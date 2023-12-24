import pandas as pd
import numpy as np
import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os
from typing import Optional, Tuple, Union
import random
# from ditto.model import DittoModel,DittoDataset,load_model,to_str,classify,train,simple_train
from train import ValidDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
# from xgboost import XGBClassifier
import time
import warnings
from types import SimpleNamespace
import re
warnings.filterwarnings("ignore")


def find_indexes(lst):
    indexes = []
    for i in range(len(lst)):
        if lst[i] == 1:
            indexes.append(i+1) ## Note we should promise i>0 for feature selection
    return indexes

def get_reward(test_data,cols,relation_dict,model_output,epoch,mutual_info,da,lm,data_name):
    args = SimpleNamespace(num_iter = 1,
                     device='cuda',
                     batch_size=64,
                     max_length=128,
                     learning_rate=3e-5,
                     n_epochs=5,
                     save_model=True,
                     model_name_or_path=lm,
                     fp16=True,
                     add_token=True,
                     dynamic_dataset=-1,
                     pseudo_label_method='uncertainty',
                     model_type='roberta',
                     template_no=0,
                     one_word=False,
                     teacher_epochs=10,
                     student_epochs=10,
                     lr=2e-5,
                     self_training=False,
                     )
    if data_name!='amazon-google':
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
    if(data_name=='persons'):
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
    elif(data_name=='imdb' or data_name=='imdb-1hop'):
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
    elif(data_name=='amazon-google'):
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
    elif(data_name=='dblp'):
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
    elif(data_name=='imdb-3hop'):
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
    elif(data_name=='persons-3hop'):
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
    elif(data_name=='amazon-google-3hop' or data_name=='wdc'):
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
    mutual_info_index = np.where(np.array(cols) == 1)[0]
    if(len(mutual_info_index)>0):
        mutual_info_score = np.mean(mutual_info[mutual_info_index])
    else:
        mutual_info_score = np.mean(mutual_info)
    cols = find_indexes(cols)
    random_numbers = np.random.choice(len(test_data), 256, replace=False)
    features = RL_feature_selection(cols)


    enrich_base_valid_select = test_data.iloc[random_numbers,features]

    if enrich_base_valid_select.shape[1]>9:
        completeness = 1 - sum(enrich_base_valid_select.iloc[:,9:].eq('').sum()) / (enrich_base_valid_select.shape[0] * (enrich_base_valid_select.shape[1] - 9))
    else:
        completeness = 0

    enrich_base_valid_output = enrich_base_valid_select.apply(ditto_transfer_extend,axis=1,result_type='expand')
    reward = ValidDataLoader(args,enrich_base_valid_output,model_output)
    if epoch==0:
        reward_output = reward + completeness + mutual_info_score
    else:
        reward_output = reward + completeness + mutual_info_score
    print(reward,completeness,mutual_info_score)
    return np.float32(reward_output)

class Environment_SchemaEnr(gym.Env):
    def __init__(self, mask ,relation_dict,mutual_info,state_path,test_data,data_name,feature_num,max_episode_steps: int = 100,device_id: int = 3,epoch: int = 0, model_input = False,task: str = 'wiki', max_path: int=5,da=None,lm: str = 'roberta'):
        self.state_path = str(state_path)
        self.lm = lm
        self.data_name = data_name
        self.max_path = max_path
        self.da = da
        self.epoch = epoch
        self.mask = mask
        self.device_id = device_id
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % str(self.device_id)
        self.max_episode_steps = 150
        total_feature_num = feature_num
        self.steps = 0
        # self.max_episode_steps = max_episode_steps
        self.max_episode_steps = 150
        self.total_feature_num = total_feature_num
        self.model = model_input
        self.mutual_info = mutual_info
        self.action_space = Discrete(total_feature_num)
        self.n_action = self.action_space.n
        self.observation_space = Box(0,total_feature_num-1,shape=(total_feature_num,), dtype=np.int64)
        self.state = [0]*total_feature_num
        self.test_data = test_data
        self.state_memory = []
        self.reward_memory = []
        self.relation_dict = relation_dict
        self.task = task
        if not os.path.exists('%s/data/state/%s' % (self.state_path,str(self.task))):
            os.makedirs('%s/data/state/%s' % (self.state_path,str(self.task)))
    def reset(
        self,
        *,
        seed= None,
        options= None,
    ):
        init = [0] * self.total_feature_num
        self.state = np.array(init)
        return self.state,{}
    def step(self,action):
        start = time.time()
        a = action
        temp = []
        act = [0]*a + [1] + [0]*(self.total_feature_num-a-1)
        for i in range(len(act)):
            temp.append((act[i]+self.state[i])%2)
        self.state = np.array(temp)
        if temp not in self.state_memory:
            reward = get_reward(self.test_data,self.state,self.relation_dict,self.model,self.epoch,self.mutual_info,self.da,self.lm,self.data_name)
            self.state_memory.append(temp)
            self.reward_memory.append(reward)
        else:
            index = self.state_memory.index(temp)
            reward = self.reward_memory[index]
        end = time.time()

        if sum(self.state) >= self.max_path: 
            done = True
        else:
            done = False
        terminated = done
        self.steps += 1
        truncated = False
        if(self.steps % 10 ==0):
            timestamp = int(time.time())
            np.save('%s/data/state/%s/state_memory_%s.npy' % (self.state_path,str(self.task),str(self.epoch)),self.state_memory)
            np.save('%s/data/state/%s/reward_memory_%s.npy' % (self.state_path,str(self.task),str(self.epoch)),self.reward_memory)
        return self.state,reward,terminated,truncated,{}
    def valid_action_mask(self):
        return self.mask