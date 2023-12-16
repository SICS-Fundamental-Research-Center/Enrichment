## Enriching Relations with Additional Attributes for ER
### Requirements
Basic requirements are listed in requirements.txt
Additionally, you need to install 
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex (fp16 training)
* tensorflow
for the capability of [Ditto](https://github.com/megagonlabs/ditto)/ [PromptEM](https://github.com/ZJU-DAILY/PromptEM)
/[L2X](https://github.com/Jianbo-Lab/L2X)

#### Prepare the RL Env
* After the installation of gymnasium, create `gymnasium/envs/custom` folder in your pip package location of gymnasium. e.g. `/home/user/anaconda3/lib/python3.9/site-packages/gymnasium/envs/custom`. 
* Then, put all the files in folder `register_env` to `gymnasium/envs/custom`, also put `ditto` and `PromptEM` folder into the same location.
* Open `gymnasium/envs/__init__.py`, and add the following register info:
```
register(
    id="Enrich-SchemaEnr",
    entry_point="gymnasium.envs.custom.enrich_SchemaEnr:Environment_SchemaEnr",
    max_episode_steps=100,
    reward_threshold=0.96,
)
register(
    id="Enrich-SchemaEnr-PromptEM",
    entry_point="gymnasium.envs.custom.enrich_SchemaEnr_PromptEM:Environment_SchemaEnr",
    max_episode_steps=100,
    reward_threshold=0.96,
)
register(
    id="Enrich-AutoFeature-PromptEM",
    entry_point="gymnasium.envs.custom.enrich_AutoFeature_PromptEM:Environment_SchemaEnr",
    max_episode_steps=100,
    reward_threshold=0.96,
)
register(
    id="Enrich-AutoFeature",
    entry_point="gymnasium.envs.custom.enrich_AutoFeature:Environment_SchemaEnr",
    max_episode_steps=100,
    reward_threshold=0.96,
)
```
### Training with Ditto/Ditto-aug
```
python enrich_train_RL.py --data_name amazon-google --model ditto-aug --method SchemaEnr --max_path 5 --task amazon_google_SchemaEnr_da --lm roberta-base --base_model model/imdb_enrich/model.pt
```

The meaning of the flags:


- `--data_name`: the name of the dataset. options: `["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"]`
- `--model`: the selection of downstream ER model. options: `["ditto", "ditto-aug"]`
- `--update`: the flag to enable updating the A_er model during training. The default is True.
- `--method`: the RL method to select. options: `["SchemaEnr","AutoFeature"]`
- `--max_path`: decide the max attributes to select during RL. Default is 5.
- `--task`: the name of current task.
- `--lm`: the path of the local language model folder(currently support Roberta). The default is Roberta-base
- `--base_model`: the file path of ER model training on the base attributes. The format is .ckpt.
- `--epoch`: Training epoch for updating the A_er per episode. 

### Training with PromptEM
```
cd PromptEM
python enrich_train_RL_PromptEM.py --data_name amazon-google --method SchemaEnr --max_path 5 --task amazon_google_SchemaEnr_PromptEM --lm roberta-base --base_model model/amazon-google-base-PromptEM.ckpt
```

The meaning of the flags:


- `--data_name`: the name of the dataset. options: `["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"]`
- `--update`: the flag to enable updating the A_er model during training. The default is True.
- `--method`: the RL method to select. options: `["SchemaEnr","AutoFeature"]`
- `--max_path`: decide the max attributes to select during RL. Default is 5.
- `--task`: the name of current task.
- `--lm`: the path of the local language model folder(currently support Roberta). The default is Roberta-base
- `--base_model`: the file path of ER model training on the base attributes. The format is .ckpt.
- `--epoch`: Training epoch for updating the A_er per episode. 

### Testing for RL output with Ditto/Ditto-aug
```
python enrich_test.py --data_name amazon-google --model ditto --method SchemaEnr --max_path 5 --task amazon_google_SchemaEnr_da --lm roberta-base --save_model --epoch 16
```
The meaning of the flags:


- `--data_name`: the name of the dataset. options: `["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"]`
- `--model`: the selection of test ER model. options: `["ditto", "ditto-aug"]`
- `--method`: the feature selection method options: `["SchemaEnr","AutoFeature","Base","Random","L2X"]`. For L2X, you need to manually replace the L2X path selection with the variance l2x in enrich_test.py.
- `--max_path`: decide the max attributes to select during for Random and L2X. Default is 5.
- `--task`: the name of current task, should be the same with enrich_train_RL
- `--lm`: the path of the local language model folder(currently support Roberta). The default is Roberta-base
- `--save_model`: the flag to enable saving model, is used for base model generation. Default is False.
- `--epoch`: Training epoch for testing the given path. Default is 15. 

### Testing for RL output with PromptEM
```
cd PromptEM
python enrich_test_PromptEM.py --data_name amazon-google --model ditto --method SchemaEnr --max_path 5 --task amazon_google_SchemaEnr_da --lm roberta-base --save_model --epoch 16
```
The meaning of the flags:


- `--data_name`: the name of the dataset. options: `["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"]`
- `--method`: the feature selection method options: `["SchemaEnr","AutoFeature","Base","Random","L2X"]`. For L2X, you need to manually replace the L2X path selection with the variance l2x in enrich_test.py.
- `--max_path`: decide the max attributes to select during for Random and L2X. Default is 5.
- `--task`: the name of current task, should be the same with enrich_train_RL
- `--lm`: the path of the local language model folder(currently support Roberta). The default is Roberta-base
- `--save_model`: the flag to enable saving model, is used for base model generation. Default is False.
- `--epoch`: Training epoch for testing the given path. Default is 15. 

### Base Model Generation
#### Ditto/Ditto-aug
```
python enrich_test.py --data_name amazon-google --model ditto --method Base --max_path 5 --task amazon_google_base --lm roberta-base --save_model --epoch 10
```
Change the data_name for different base model of different datasets. The model is stored in `model_log/`
#### PromptEM

```
cd PromptEM
python enrich_test_PromptEM.py --data_name amazon-google --method Base --max_path 5 --task amazon_google_base --lm roberta-base --save_model --epoch 10
```
Change the data_name for different base model of different datasets. The model is stored in `PromptEM/model_log/`

#### L2X Similarity Generation
```
python l2x_similarity_generation.py --data_name amazon-google --lm all-mpnet-base-v2
```
The meaning of the flags:


- `--data_name`: the name of the dataset. options: `["persons", "imdb", "amazon-google", "dblp","imdb-1hop","imdb-3hop"]`
- `--lm`: the local path of the sentence-bert model, we use [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for the default option

Since L2X can only process feature selection in numerical data, we provide the cosine similarity for each attributes of Entity A and Entity B, based on sentence_bert embedding result.

This procedure can take up to 1h, based on different GPU, so we already provided the similarity output in data/l2x-similarity

#### L2X feature selection
```
python l2x_generation.py --data_name amazon-google --max_path 5
```
#### Result Path
* Ditto/Ditto-aug: `log`
* PromptEM: `PromptEM/log`

