import torch
import time
from torch.utils import data
from transformers import AutoTokenizer
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from apex import amp
from torch.utils import data
import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from apex import amp
from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):
        self.tokenizer = get_tokenizer(lm) ## PreTrainedTokenizerFast(name_or_path='distilbert-base-uncased', vocab_size=30522, model_max_len=512, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
        self.pairs = [] 
        self.labels = [] 
        self.max_len = max_len 
        self.size = size

        # if isinstance(path, list):
        #     lines = path
        # else:
        #     lines = open(path)

        # for line in lines:
        #     s1, s2, label = line.strip().split('\t')
        #     self.pairs.append((s1, s2))
        #     self.labels.append(int(label))
        if isinstance(path, pd.DataFrame):
            # 处理DataFrame中的内容
            for index, row in path.iterrows():
                s1 = row[0]
                s2 = row[1]
                label = row[2]
                self.pairs.append((s1, s2))
                self.labels.append(int(label))
        else:
            # 处理文件内容
            if isinstance(path, list):
                lines = path
            else:
                lines = open(path)

            for line in lines:
                s1, s2, label = line.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right 其中101代表[cls]，102代表[sep]。由于add_special_tokens的默认参数为True，所以中间拼接会有连接词[sep]，‘token_type_ids’:区分两个句子的编码（上句全为0，下句全为1）
        x = self.tokenizer.encode(text=left, 
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True) ## tensorflow版本（'tf'）和pytorch版本（'pt'）
        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split('[SEP]')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)
class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            


            print('LM Model 1')
            self.bert = AutoModel.from_pretrained(lm_mp[lm]) ## 加载预训练模型
        else:
            print('LM Model 2')
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()
def load_model( path, lm, use_gpu, fp16=True):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models
    # checkpoint = os.path.join(path, task, 'model.pt')
    # if not os.path.exists(checkpoint):
        # raise ModelNotFoundError(checkpoint)

    # configs = json.load(open('configs.json'))
    # configs = {conf['name'] : conf for conf in configs}
    # config = configs[task]
    # config_list = [config]

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    model = DittoModel(device=device, lm=lm)

    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])
    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return  model
def load_model_update( path, lm, use_gpu, lr,fp16=True):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models
    # checkpoint = os.path.join(path, task, 'model.pt')
    # if not os.path.exists(checkpoint):
        # raise ModelNotFoundError(checkpoint)

    # configs = json.load(open('configs.json'))
    # configs = {conf['name'] : conf for conf in configs}
    # config = configs[task]
    # config_list = [config]

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    model = DittoModel(device=device, lm=lm)

    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])
    optimizer = AdamW(model.parameters(), lr=lr)
    model = model.to(device)

    if fp16 and 'cuda' in device:
        model,optimizer = amp.initialize(model,optimizer, opt_level='O2')

    return  model,optimizer
def to_str(ent1, ent2,  max_len=256):
    """Serialize a pair of data entries

    Args:
        ent1 (Dictionary): the 1st data entry
        ent2 (Dictionary): the 2nd data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    content = ''
    for ent in [ent1, ent2]:
        if isinstance(ent, str):
            content += ent
        else:
            for attr in ent.keys():
                content += 'COL %s VAL %s ' % (attr, ent[attr])
        content += '\t'

    content += '0'


    new_ent1, new_ent2, _ = content.split('\t')

    return new_ent1 + '\t' + new_ent2 + '\t0'
def classify(sentence_pairs, model,
             lm='distilbert',
             max_len=256,
             threshold=None):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        list of float: the scores of the pairs
    """
    if(isinstance(sentence_pairs,DittoDataset)):
        dataset = sentence_pairs
    else:
        inputs = sentence_pairs
        # print('max_len =', max_len)
        dataset = DittoDataset(inputs,
                            max_len=max_len,
                            lm=lm)
    # print(dataset[0])
    
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=len(dataset),
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)

    # prediction
    all_probs = []
    all_logits = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            x, _ = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_logits += logits.cpu().numpy().tolist()

    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]
    return pred, all_logits

def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)

        loss = criterion(prediction, y.to(model.device)) ## 把标签加载到GPU上面

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 50 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss
def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th
def train(trainset, validset, testset, hp):
    
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        start = time.time()
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                # print(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")
        end = time.time()
        # print(end-start)
        # logging
        # scalars = {'f1': dev_f1,
        #            't_f1': test_f1}
def simple_train(trainset, validset, testset, hp):
    
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        start = time.time()
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)


        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                # print(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")
        end = time.time()
        print(end-start)
        # logging
        # scalars = {'f1': dev_f1,
        #            't_f1': test_f1}
    # if hp.save_model:
    # # create the directory if not exist
    #     directory = os.path.join(hp.logdir, hp.task)
    #     # print(directory)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    #     # save the checkpoints for each component
    #     ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
    #     ckpt = {'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'scheduler': scheduler.state_dict(),
    #             'epoch': epoch}
    #     torch.save(ckpt, ckpt_path)
    return model
def simple_train_update(trainset, validset, testset, hp,optimizer, model_input=False):
    
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(isinstance(model_input,DittoModel)):
        model = model_input.cuda()
    elif os.path.isfile(model_input):
        model,optimizer = load_model_update(model_input,'roberta-base',use_gpu=True,lr=hp.lr,fp16=True)
        # model = model.cuda()
    else:
        model = DittoModel(device=device,
                        lm=hp.lm,
                        alpha_aug=hp.alpha_aug)
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)

    # if hp.fp16:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        start = time.time()
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)


        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                # print(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")
        end = time.time()
        print(end-start)
        # logging
        # scalars = {'f1': dev_f1,
        #            't_f1': test_f1}
    # if hp.save_model:
    # # create the directory if not exist
    #     directory = os.path.join(hp.logdir, hp.task)
    #     # print(directory)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    #     # save the checkpoints for each component
    #     ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
    #     ckpt = {'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'scheduler': scheduler.state_dict(),
    #             'epoch': epoch}
    #     torch.save(ckpt, ckpt_path)
    return model