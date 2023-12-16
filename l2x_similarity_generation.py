import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
import re
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str,
                    choices=["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"], default="persons")
parser.add_argument("--lm", type=str,default="sentence-transformers_all-mpnet-base-v2")
main_args = parser.parse_args()

model = SentenceTransformer(main_args.lm)
model = model.to('cuda')
def calculate_similarity(matrix1, matrix2):

    index = faiss.IndexFlatL2(matrix2.shape[1])
    index.add(matrix2)


    distances, _ = index.search(matrix1, 1)
    similarities = 1 / (1 + distances)
    result = similarities.reshape(-1, 1)

    return result
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
if(main_args.data_name=='imdb-1hop'):
    data_encode_train = enrich_train.iloc[:,9:45].astype(str).values.flatten()
    data_encode_valid = enrich_valid.iloc[:,9:45].astype(str).values.flatten()
elif(main_args.data_name=='imdb-3hop'):
    data_encode_train = enrich_train.iloc[:,5:].astype(str).values.flatten()
    data_encode_valid = enrich_valid.iloc[:,5:].astype(str).values.flatten()
else:
    data_encode_train = enrich_train.iloc[:,9:].astype(str).values.flatten()
    data_encode_valid = enrich_valid.iloc[:,9:].astype(str).values.flatten()
embedding_matrix_train = model.encode(data_encode_train,show_progress_bar=True)
embedding_matrix_valid = model.encode(data_encode_valid,show_progress_bar=True)
embedding_matrix_train_reshape = embedding_matrix_train.reshape((len(enrich_train),-1,768))
embedding_matrix_valid_reshape = embedding_matrix_valid.reshape((len(enrich_valid),-1,768))
feature_num = int(embedding_matrix_train.shape[1]/2)
similarity_train = np.zeros((len(enrich_train),feature_num))
similarity_valid = np.zeros((len(enrich_valid),feature_num))
for i in tqdm(range(feature_num)):
    result = calculate_similarity(np.ascontiguousarray(embedding_matrix_train_reshape[:,2*i]),np.ascontiguousarray(embedding_matrix_train_reshape[:,2*i+1]))
    similarity_train[:,i] = result.reshape((-1))
for i in tqdm(range(feature_num)):
    result = calculate_similarity(np.ascontiguousarray(embedding_matrix_valid_reshape[:,2*i]),np.ascontiguousarray(embedding_matrix_valid_reshape[:,2*i+1]))
    similarity_valid[:,i] = result.reshape((-1))
    
similarity_train = pd.DataFrame(similarity_train).to_csv('data/l2x-similarity/%s_train.csv' % main_args.data_name)
similarity_valid = pd.DataFrame(similarity_valid).to_csv('data/l2x-similarity/%s_valid.csv' % main_args.data_name)