from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
import gymnasium
from ditto.model import DittoModel,DittoDataset,load_model,to_str,classify,train,simple_train,simple_train_update,load_model_update
import argparse
import re
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from types import SimpleNamespace
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sklearn.metrics import f1_score
from stable_baselines3 import DQN
import time
from datetime import datetime
time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# time_stamp = time.time()
import os
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str,
                    choices=["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop","persons-3hop","amazon-google-3hop","wdc","wdc-K=1","wdc-K=3","wdc-K=5","wdc-K=7","wdc-K=9","persons-1hop"], default="persons")
parser.add_argument("--model", type=str,
                    choices=["ditto", "ditto-aug"], default="ditto")
parser.add_argument("--update", action="store_true", default=False)
parser.add_argument("--method", type=str,
                    choices=["Base","Random","L2X","SchemaEnr", "AutoFeature", "Full", "MI"], default="SchemaEnr")
parser.add_argument("--max_path", type=int, default=5)
parser.add_argument("--task", type=str,default="persons_SchemaEnr")
parser.add_argument("--lm", type=str,default="roberta") ## If you have a custom folder, replace with the path
parser.add_argument("--base_model", type=str,default="model/wiki_base/person_enrich/model.pt")
# parser.add_argument("--OneHop", action="store_true", default=False)
# parser.add_argument("--ThreeHop", action="store_true", default=False)
parser.add_argument("--save_model", action="store_true", default=False)
parser.add_argument("--epoch", type=int, default=15)
parser.add_argument("--device", type=str, default=0)


action_space_dict = {'persons':491,
                     'imdb':48,
                     'amazon-google':318,
                     'dblp':123,
                     'imdb-1hop':18,
                     'imdb-3hop':84,
                     'persons-3hop':768,
                     'persons-1hop':212,
                     'amazon-google-3hop':446,
                     'wdc':285,
                     "wdc-K=1":285,
                     'wdc-K=3':285,
                     'wdc-K=5':285,
                     'wdc-K=7':285,
                     'wdc-K=9':285,}

main_args = parser.parse_args()
state_path = os.getcwd()
max_path = main_args.max_path
task=main_args.task
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % main_args.device

logging.basicConfig(filename='log/%s-%s.txt' % (str(task),str(time_stamp)), level=logging.DEBUG)
logger = logging.getLogger('my_logger')
file_handler = logging.FileHandler('log/%s-%s.txt' % (str(task),str(time_stamp)))
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)





# env_choose = env_dict['%s_%s' % (main_args.data_name,main_args.method)]
action_space = action_space_dict[main_args.data_name]
# if(main_args.OneHop):
#     action_space = action_space_dict['imdb-1hop']
# if(main_args.ThreeHop):
#     action_space = action_space_dict['imdb-3hop']

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
    
if main_args.method not in ["SchemaEnr", "AutoFeature"]:
    episode = [0]
else:
    episode = [0,1,2,3,4,5,6,7,8,9]
    
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
elif(main_args.data_name=='persons-3hop'):
    mask_array = np.load('data/person-3hop/wiki_mask.npy')
    mutual_info = np.load('data/person-3hop/mutual_info_norm.npy')
    mutual_info = np.mean(mutual_info,axis=0)
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/person-3hop/enrich_3hop_train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/person-3hop/enrich_3hop_valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/person-3hop/enrich_3hop_test.csv',index_col=0).fillna('')
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
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 2-hop
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col) 
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col) 
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='persons-1hop'):
    mask_array = np.load('data/person-3hop/wiki_mask.npy')[:212]
    mutual_info = np.load('data/person-3hop/mutual_info_norm.npy')
    mutual_info = np.mean(mutual_info,axis=0)[:212]
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/person-3hop/enrich_3hop_train.csv',index_col=0).fillna('').iloc[:,:433]
    enrich_valid = pd.read_csv('data/person-3hop/enrich_3hop_valid.csv',index_col=0).fillna('').iloc[:,:433]
    enrich_test = pd.read_csv('data/person-3hop/enrich_3hop_test.csv',index_col=0).fillna('').iloc[:,:433]
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
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 2-hop
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col) 
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col) 
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='amazon-google-3hop'):
    mask_array = np.load('data/amazon-google-3hop/amazon_mask.npy')
    mutual_info = np.load('data/amazon-google-3hop/mutual_info_norm.npy')
    mutual_info = np.mean(mutual_info,axis=0)
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/amazon-google-3hop/train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/amazon-google-3hop/valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/amazon-google-3hop/test.csv',index_col=0).fillna('')
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
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 2-hop
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col) 
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col) 
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name=='wdc'):
    mask_array = np.load('data/wdc-shoes/wdc_mask.npy')
    mutual_info = np.load('data/wdc-shoes/wdc_mutual_info_norm.npy')
    mutual_info = np.mean(mutual_info,axis=0)
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/wdc-shoes/train.csv',index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/wdc-shoes/valid.csv',index_col=0).fillna('')
    enrich_test = pd.read_csv('data/wdc-shoes/test.csv',index_col=0).fillna('')
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
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 2-hop
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col) 
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col) 
        return ent_1_text,ent_2_text,row['label']
elif(main_args.data_name.__contains__('wdc-')):
    mask_array = np.load('data/%s/wdc_mask.npy' % main_args.data_name)
    mutual_info = np.load('data/%s/wdc_mutual_info_norm.npy'% main_args.data_name)
    mutual_info = np.mean(mutual_info,axis=0)
    relation_dict = np.load('relation_dict.npy',allow_pickle=True).item()
    enrich_train = pd.read_csv('data/%s/train.csv'% main_args.data_name,index_col=0).fillna('')
    enrich_valid = pd.read_csv('data/%s/valid.csv'% main_args.data_name,index_col=0).fillna('')
    enrich_test = pd.read_csv('data/%s/test.csv'% main_args.data_name,index_col=0).fillna('')
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
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col)
                else: ## 2-hop
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_1_text += 'COL %s VAL %s ' % (relation,col) 
            elif(index.__contains__('_b')):
                path_str = index[:-2]
                if(not path_str.__contains__('_')):
                    relation = relation_dict[int(index[:-2])] 
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                elif(path_str.count('_')==1):
                    relations = path_str.split('_')
                    relation = '%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col)
                else:
                    relations = path_str.split('_')
                    relation = '%s_%s_%s' % (relation_dict[int(relations[0])],relation_dict[int(relations[1])],relation_dict[int(relations[2])])
                    ent_2_text += 'COL %s VAL %s ' % (relation,col) 
        return ent_1_text,ent_2_text,row['label']

hp_simple = SimpleNamespace(task=main_args.task,
                     batch_size=128,
                     max_len=128,
                     lr=3e-5,
                     n_epochs=main_args.epoch,
                     save_model=main_args.save_model,
                     logdir="model_log/",
                     lm=main_args.lm,
                     fp16=True,
                     alpha_aug=0.8,
                     da=da)
for epoch_count in episode:
    if(main_args.method=='L2X'):
        l2x = np.array([56,354,281,90,239]) + 1 ## Toy Example for persons-L2X-m=5, replace [56,354,281,90,239] with L2X result array
        feature = (RL_feature_selection(l2x))
    elif(main_args.method=='Base'):
        l2x = []
        feature = (RL_feature_selection(l2x)) 
    elif(main_args.method=='Random'):
        l2x = np.random.choice(action_space, max_path, replace=False) + 1
        feature = (RL_feature_selection(l2x))
    elif(main_args.method=='Full'):
        l2x = np.random.choice(action_space, min(50,action_space), replace=False) + 1
        feature = (RL_feature_selection(l2x))
    elif(main_args.method=='MI'):
        l2x = np.argsort(-mutual_info)[:main_args.max_path] + 1
        print(l2x)
        feature = (RL_feature_selection(l2x))
        print(feature)
    else:
        state_memory = np.load('data/state/%s/state_memory_%s.npy' % (str(task),str(epoch_count)))
        reward_memory = np.load('data/state/%s/reward_memory_%s.npy' % (str(task),str(epoch_count)))
        feature = RL_feature_selection(find_indexes(state_memory[np.argmax(reward_memory)]))
    enrich_base_train_output = enrich_train.iloc[:,feature].progress_apply(ditto_transfer_extend,axis=1,result_type='expand')
    enrich_base_valid_output = enrich_valid.iloc[:,feature].progress_apply(ditto_transfer_extend,axis=1,result_type='expand')
    enrich_base_test_output = enrich_test.iloc[:,feature].progress_apply(ditto_transfer_extend,axis=1,result_type='expand')
    train_dataset = DittoDataset(enrich_base_train_output,max_len=128,lm = main_args.lm,da=da)
    valid_dataset = DittoDataset(enrich_base_valid_output,max_len=128,lm = main_args.lm)
    test_dataset = DittoDataset(enrich_base_test_output,max_len=128,lm = main_args.lm)
    model_output = simple_train(train_dataset,valid_dataset,test_dataset,hp_simple)
    predict = classify(test_dataset,model=model_output,lm=main_args.lm,max_len=128,threshold=0.5)
    result_f1 = f1_score(y_pred=predict[0],y_true=enrich_base_test_output.iloc[:,-1])
    print(f1_score(y_pred=predict[0],y_true=enrich_base_test_output.iloc[:,-1]))
    logger.info('episode_%s Task: %s f1-score: %s, path: %s' % (str(epoch_count),task,str(result_f1),str(feature)))