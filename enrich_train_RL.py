from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
import gymnasium
from ditto.model import DittoModel,DittoDataset,load_model,to_str,classify,train,simple_train,simple_train_update,load_model_update
import argparse
import re
from types import SimpleNamespace
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sklearn.metrics import f1_score
from stable_baselines3 import DQN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str,
                    choices=["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"], default="persons")
parser.add_argument("--model", type=str,
                    choices=["ditto", "ditto-aug"], default="ditto")
parser.add_argument("--update", action="store_true", default=True)
parser.add_argument("--method", type=str,
                    choices=["SchemaEnr", "AutoFeature"], default="SchemaEnr")
parser.add_argument("--max_path", type=int, default=5)
parser.add_argument("--task", type=str,default="persons_SchemaEnr")
parser.add_argument("--lm", type=str,default="roberta") ## If you have a custom folder, replace with the path
parser.add_argument("--base_model", type=str,default="model/wiki_base/person_enrich/model.pt")
parser.add_argument("--OneHop", action="store_true", default=False)
parser.add_argument("--ThreeHop", action="store_true", default=False)
parser.add_argument("--epoch", type=int, default=5)
# env_dict = {'persons_SchemaEnr':"Enrich-wiki",
#             'persons_AutoFeature':"Enrich-wiki-dqn",
#             'imdb_SchemaEnr':"Enrich-imdb",
#             'imdb_AutoFeature':"Enrich-imdb-dqn",
#             'amazon-google_SchemaEnr':"Enrich-amazon",
#             'amazon-google_AutoFeature':"Enrich-amazon-dqn",
#             'dblp_SchemaEnr':"Enrich-dblp",
#             'dblp_AutoFeature':"Enrich-dblp-dqn",
#             'imdb_SchemaEnr_1hop':"Enrich-imdb-1hop",
#             'imdb_AutoFeature_1hop':"Enrich-imdb-1hop-dqn",
#             'imdb_SchemaEnr_3hop':"Enrich-imdb-3hop",
#             'imdb_AutoFeature_3hop':"Enrich-imdb-3hop-dqn",
#     }
action_space_dict = {'persons':491,
                     'imdb':48,
                     'amazon-google':318,
                     'dblp':123,
                     'imdb-1hop':18,
                     'imdb-3hop':84}





main_args = parser.parse_args()
state_path = os.getcwd()
max_path = main_args.max_path
task=main_args.task
# env_choose = env_dict['%s_%s' % (main_args.data_name,main_args.method)]
# if(main_args.OneHop):
#     env_choose = env_dict['%s_%s_1hop' % (main_args.data_name,main_args.method)]
# elif(main_args.ThreeHop):
#     env_choose = env_dict['%s_%s_3hop' % (main_args.data_name,main_args.method)]
    
# if(main_args.OneHop):
#     action_space = action_space_dict['imdb-1hop']
# if(main_args.ThreeHop):
#     action_space = action_space_dict['imdb-3hop']
action_space = action_space_dict[main_args.data_name]

if main_args.data_name!='amazon-google':
    def RL_feature_selection(selection_list):
        list = [0,1,2,3,4,5,6,7,8]
        for s in selection_list: 
            list.append(2*(s-1)+9)
            list.append(2*s+8)
        return list
else:
    def RL_feature_selection(selection_list):
        list = [0,1,2,3,4]
        for s in selection_list: 
            list.append(2*s+3)
            list.append(2*s+4)
        return list
def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.valid_action_mask()
def find_indexes(lst):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == 1:
            indexes.append(i+1) ## Note we should promise i>0 for feature selection
    return indexes

def find_state_memory_indices(state_memory, reward_memory):
    row_sums = np.sum(state_memory, axis=1)
    indices_with_rewards = list(enumerate(row_sums, 0))
    indices_with_rewards.sort(key=lambda x: x[1], reverse=True)
    result = {
        'sum_5': [],
        'sum_4': [],
        'sum_3': [],
        'sum_2': [],
        'sum_1': []
    }
    for index_num, row_sum in indices_with_rewards:
        index_col = state_memory[index_num]
        index = RL_feature_selection(find_indexes(index_col))
        if len(result['sum_5']) < 20 and row_sum == 5:
            result['sum_5'].append(index)
        elif len(result['sum_4']) < 20 and row_sum == 4:
            result['sum_4'].append(index)
        elif len(result['sum_3']) < 10 and row_sum == 3:
            result['sum_3'].append(index)
        elif len(result['sum_2']) < 10 and row_sum == 2:
            result['sum_2'].append(index)
        elif len(result['sum_1']) < 10 and row_sum == 1:
            result['sum_1'].append(index)
    return result

if main_args.model=='ditto-aug':
    da = 'all'
else:
    da = None
    
if(main_args.data_name=='persons'):
    mask_array = np.load('data/person/wiki_mask.npy')
    mutual_info = np.load('data/person/mutual_info_norm.npy')
    mutual_info = np.mean(mutual_info,axis=0)
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/person/enrich_2hop_train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/person/enrich_2hop_valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/person/enrich_2hop_test.csv',index_col=0).fillna('')
    def ditto_transfer_extend(row):
        ent_1 = row[1:4]
        ent_1_text = ''
        for index,col in ent_1.iteritems():
            ent_1_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        ent_2 = row[5:8]
        ent_2_text = ''
        for index,col in ent_2.iteritems():
            ent_2_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        for index,col in row[9:].iteritems():
            if(index.__contains__('_a')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='imdb' or main_args.data_name=='imdb-1hop'):
    mask_array = np.load('model/imdb_mask.npy')
    if(main_args.data_name=='imdb-1hop'):
        mask_array = mask_array[:18]
    mutual_info = np.load('data/imdb/imdb_mutual_info.npy')
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/imdb/enrich_extend_imdb_2hop_train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/imdb/enrich_extend_imdb_2hop_valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/imdb/enrich_extend_imdb_2hop_test.csv',index_col=0).fillna('')
    def ditto_transfer_extend(row):
        ## Process Entity A
        ent_1 = row[1:4]
        ent_1_text = ''
        for index,col in ent_1.iteritems():
            # print(index)
            ent_1_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        ent_2 = row[5:8]
        ent_2_text = ''
        for index,col in ent_2.iteritems():
            # print(index)
            ent_2_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        # if(len(row)>9):
        #     total_length = len(row) - 9
        #     for t in range(int(total_length/2) ): ## Search All attributes by pairs
        for index,col in row[9:].iteritems():
            if(index.__contains__('_a')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='amazon-google'):
    mask_array = np.load('data/amazon-google/amazon_mask.npy')
    mutual_info = np.load('data/amazon-google/amazon_mutual_info.npy')
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/amazon-google/train_extend.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/amazon-google/valid_extend.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/amazon-google/test_extend.csv',index_col=0).fillna('')
    def ditto_transfer_extend(row):
        ## Process Entity A
        ent_1 = row[1:2]
        ent_1_text = ''
        for index,col in ent_1.iteritems():
            # print(index)
            ent_1_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        ent_2 = row[3:4]
        ent_2_text = ''
        for index,col in ent_2.iteritems():
            # print(index)
            ent_2_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        # if(len(row)>9):
        #     total_length = len(row) - 9
        #     for t in range(int(total_length/2) ): ## Search All attributes by pairs
        for index,col in row[5:].iteritems():
            if(index.__contains__('_a')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] ## relation a, cut for length limit
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] ## relation a, cut for length limit
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)         
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='dblp'):
    mask_array = np.load('model/dblp_mask.npy')
    mutual_info = np.load('data/dblp/dblp_mutual_info.npy')
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/dblp/dblp_train_extend.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/dblp/dblp_valid_extend.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/dblp/dblp_test_extend.csv',index_col=0).fillna('')
    def relation_process(input):
        if(input.__contains__('#')):
            output = input.split('#')[-1]
        else:
            output = input.split('/')[-1]
        return output
    def extract_elements(input_str):
        if(input_str.__contains__('<')):
            elements = re.findall(r'<(.*?)>', input_str)
            result = [element for element in elements if not element.endswith(("_a", "_b", "_0", "_1", "_2"))]
            return relation_process(result[0]) if len(result)==1 else relation_process(result[0]) + '_' + relation_process(result[1])
        else:
            return input_str.replace('_a','').replace('_b','')
    def ditto_transfer_extend(row):
        ent_1 = row[1:4]
        ent_1_text = ''
        for index,col in ent_1.iteritems():
            # print(index)
            ent_1_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        ent_2 = row[5:8]
        ent_2_text = ''
        for index,col in ent_2.iteritems():
            # print(index)
            ent_2_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        for index,col in row[9:].iteritems():
            if(index.__contains__('_a')):
                path_str = index
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_1_text += 'COL %s VAL %s ' % (extract_elements(relation),col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_1_text += 'COL %s VAL %s ' % (extract_elements(relation),col)
            elif(index.__contains__('_b')):
                path_str = index
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_2_text += 'COL %s VAL %s ' % (extract_elements(relation),col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_2_text += 'COL %s VAL %s ' % (extract_elements(relation),col)
                
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='imdb-3hop'):
    mask_array = np.load('data/imdb/imdb_3hop_mask.npy')
    mutual_info = np.load('data/imdb/imdb_mutual_info_3hop.npy')
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/imdb/imdb_3hop_train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/imdb/imdb_3hop_valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/imdb/imdb_3hop_test.csv',index_col=0).fillna('')
    def ditto_transfer_extend(row):
        ## Process Entity A
        ent_1 = row[1:4]
        ent_1_text = ''
        for index,col in ent_1.iteritems():
            # print(index)
            ent_1_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        ent_2 = row[5:8]
        ent_2_text = ''
        for index,col in ent_2.iteritems():
            # print(index)
            ent_2_text += 'COL %s VAL %s ' % (index.replace('_a','').replace('_b',''),col)
        # if(len(row)>9):
        #     total_length = len(row) - 9
        #     for t in range(int(total_length/2) ): ## Search All attributes by pairs
        for index,col in row[9:].iteritems():
            if(index.__contains__('_a')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1): ## 2-hop table
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 3-hop table
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relations[0],relations[1],relations[2])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = index[:-2] ## relation a, cut for length limit
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1): ## 2-hop table
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relations[0],relations[1])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 3-hop table
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relations[0],relations[1],relations[2])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
        return ent_1_text,ent_2_text,row['label']
hp_simple = SimpleNamespace(task=main_args.task,
                     batch_size=128,
                     max_len=128,
                     lr=3e-5,
                     n_epochs=main_args.epoch,
                     save_model=False,
                     logdir="model_log/",
                     lm=main_args.lm,
                     fp16=True,
                     alpha_aug=0.8,
                     da=da)

ditto_model,ditto_optimizer = load_model_update(main_args.base_model,lm=main_args.lm,lr=hp_simple.lr,use_gpu=True,fp16=True) ## Load Base Model
for epoch_count in range(10):
    if(main_args.method=='SchemaEnr'): 
        env = gymnasium.make('Enrich-SchemaEnr',mask=mask_array,relation_dict=relation_dict,mutual_info=mutual_info,state_path=state_path,test_data=enrich_valid,data_name=main_args.data_name,feature_num=action_space, max_episode_steps=100,device_id=2,epoch=epoch_count,model_input = ditto_model,task=hp_simple.task,max_path = max_path ,lm=main_args.lm)
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1,n_steps=40,n_epochs=10)
        print('Epoch:%s' % str(epoch_count))
        model.learn(total_timesteps=400,progress_bar=True)
    elif(main_args.method=='AutoFeature'):
        env = gymnasium.make('Enrich-AutoFeature',mask=mask_array,relation_dict=relation_dict,mutual_info=mutual_info,state_path=state_path,test_data=enrich_valid,data_name=main_args.data_name,feature_num=action_space, max_episode_steps=100,device_id=2,epoch=epoch_count,model_input = ditto_model,task=hp_simple.task,max_path = max_path ,lm=main_args.lm)
        model = DQN("MlpPolicy", env, verbose=1)
        print('Epoch:%s' % str(epoch_count))
        model.learn(total_timesteps=400,progress_bar=True)
    if(main_args.update):
        state_memory = np.load('%s/data/state/%s/state_memory_%s.npy' % (state_path,main_args.task,str(epoch_count)))
        reward_memory = np.load('%s/data/state/%s/reward_memory_%s.npy' % (state_path,main_args.task,str(epoch_count)))
        output = find_state_memory_indices(state_memory, reward_memory)
        count = sum(len(value) for value in output.values())
        count_value = [item for sublist in output.values() for item in sublist]
        header = ['0','1','label']
        update_data = pd.DataFrame(columns=header)
        update_data_valid = pd.DataFrame(columns=header)
        update_data_test = pd.DataFrame(columns=header)
        for i in range(count):
            random_numbers = np.random.choice(len(enrich_train), 64, replace=False)
            features = count_value[i]
            test_data = enrich_train.iloc[random_numbers,features]
            test_data_output = test_data.apply(ditto_transfer_extend,axis=1,result_type='expand')
            test_data_output.columns = header
            update_data = pd.concat([update_data,test_data_output])
        for i in range(count):
            random_numbers = np.random.choice(len(enrich_valid), 64, replace=False)
            features = count_value[i]
            test_data = enrich_valid.iloc[random_numbers,features]
            test_data_output = test_data.apply(ditto_transfer_extend,axis=1,result_type='expand')
            test_data_output.columns = header
            update_data_valid = pd.concat([update_data_valid,test_data_output])
        for i in range(count):
            random_numbers = np.random.choice(len(enrich_test), 64, replace=False)
            features = count_value[i]
            test_data = enrich_test.iloc[random_numbers,features]
            test_data_output = test_data.apply(ditto_transfer_extend,axis=1,result_type='expand')
            test_data_output.columns = header
            update_data_test = pd.concat([update_data_test,test_data_output])

        train_dataset = DittoDataset(update_data,max_len=128,lm = main_args.lm,da=da)
        valid_dataset = DittoDataset(update_data_valid,max_len=128,lm = main_args.lm)
        test_dataset = DittoDataset(update_data_test,max_len=128,lm = main_args.lm)
        ditto_model = simple_train_update(train_dataset,valid_dataset,test_dataset,hp_simple,model_input=ditto_model,optimizer=ditto_optimizer)
model.save(main_args.task)