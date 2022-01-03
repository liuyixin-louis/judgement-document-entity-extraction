from transformers import *
import pandas as pd
import numpy as np
import re
from torch import nn
from tqdm import tqdm
from LAC import LAC
from torch import nn
import torch
from transformers import BertTokenizer
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import re
import jieba
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import *
import torch.nn.functional as F
import joblib
from copy import deepcopy
import cn2an


# 参数
MAX_OMODEL_LENGTH = 306
MAX_JINEMODEL_LENGTH = 32


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def exact_all(x):
    res = {}
    res['amount'] = exact_ten_jine(x)
    res['deadline'] = exact_deadline(x)
    res['guarantor_maximum_guaranteed_amount'] = exact_baozheng_relation(x)
    res['mortgage_maximum_mortgage_amount'] = exact_diyawu_relation(x)
    return res

def exact_mortgage_(x):
    res = {}
    res['mortgage_maximum_mortgage_amount'] = exact_diyawu_relation(x)
    return res

def exact_ten_jine(x):
    """
    str x: 裁判文书 
    dict[list] y: 10类金额实体的列表
    """
    y = {}
    y['debt_amount'] = exact_daikuanbenjin(x)

    # 提取剩下9个金额实体
    othernine = exact_nine_jine(x)
    if othernine!=None:
        for k,v in othernine.items():
            y[k] = v
    return y

def exact_nine_jine(x):
    """
    str x: 裁判文书 
    dict[list] y: 9类金额实体的列表（不包括资本本金）
    """
    y = exact_nine_jine_model(x)
    return y

def exact_daikuanbenjin(x):
    """
    str x: 裁判文书 
    list y: 贷款本金列表
    """
    return exact_daikuanbenjin_model(x)

def exact_baozhengren(x):
    # 单个样本输入，输入为Unicode编码的字符串
    # text = u"被告广东沃帮投资有限公司、方广彬、王振豪在55560000元的最高限额内对被告广东永安贸易有限公司上述债务承担连带清偿责任；被告广东沃帮投资有限公司、方广彬、王振豪承担保证责任后，有权依照《中华人民共和国担保法》第三十一条的规定向被告广东永安贸易有限公司追偿"
    lac_result = lac.run(x)
    return [lac_result[0][i] for i in range(len(lac_result[0])) if lac_result[1][i] == 'PER' or lac_result[1][i] == 'ORG' and '银行' not in lac_result[0][i]]


def exact_deadline(x):
    """
    str x: 裁判文书 
    list y: 利息截止时间
    """
    # 规则提取
    panjue = exact_panjue(x)
    # print(panjue)
    y = [exact_time(panjue[gi.span()[0]:])[0] for gi in re.finditer('(利息计算至)|(暂计至)|(计至)|(截至)',panjue) if exact_time(panjue[gi.span()[0]:])]
    return y

def exact_daikuanbenjin_model(x):
    # print(x)
    start = ['本院认为','经审理查明','经本院审查查明','经审理查明','本院认定事实如下','经审理查明','经审理查明','本院经审理认定事实如下','经审理查明']
    start = list(set(start))
    key_text = ''
    for si in start:
        if si in x:
            res = [m.start() for m in re.finditer(si, x)]
            if len(res) == 1:
                key_text = x[res[0]:]
            else:
                # 一般都在靠中间
                original_text_length = len(x)
                mid_pos = original_text_length/2
                # 靠中间最近那个
                index = np.argmin(np.abs(np.array(res)-mid_pos),axis=0)
                pos = res[index]
                key_text = x[pos:]
    res = []
    exact_res = exact_jine_all(key_text)
    if len(exact_res)!=0:
        res = [exact_res[0][0]]
    return res

def exact_time(x):
    lac_result = lac.run(x)
    res = []
    str=''
    for i in range(len(lac_result[0])):
        if lac_result[1][i] == 'TIME':
            str+=lac_result[0][i]
        else:
            if str != '':
                res.append(str)
                str=""
    return res

# Create the BertEmbedding class
class Omodel(nn.Module):
    """Bert Model for Embedding.
    """
    def __init__(self, loss_fn=None,freeze_bert=False,bert_path=None):
        """
        @param    bert: a BertModel object
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(Omodel, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        EMBEDDING_SIZE=312
        D_in, D_out =  EMBEDDING_SIZE, 4
        self.bert  = AutoModel.from_pretrained(bert_path)

        # layernorm
        self.layernorm = nn.LayerNorm([MAX_OMODEL_LENGTH,EMBEDDING_SIZE])

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in,D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.loss = loss_fn

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        
        # Extract all the embeddingToken and layer norm
        outputs = outputs[0][:,:MAX_OMODEL_LENGTH,:]
        outputs = self.layernorm(outputs)

        # feed the outputs to classifier
        logit = self.classifier(outputs)

        return logit


# 预测
def omodel_predict(model, test_dataloader,label = True):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_pred = []
    decoded = []
    ids = []

    # For each batch in our test set...
    for batch in test_dataloader:
        #     # Load batch to GPU
        #     b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Load batch to GPU
        if label :
            id,mask,_ = tuple(t.to(device) for t in batch)
        else:
            id,mask = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(id,mask)

        all_pred.append(logits)
        ids.append(id)
    
    # Concatenate logits from each batch
    all_pred = torch.cat(all_pred, dim=0)
    ids = torch.cat(ids, dim=0)
    
    # head,tail predict
    logit_head = all_pred[:,:,:2]
    logit_tail = all_pred[:,:,2:]
    preds_head = torch.argmax(logit_head, dim=2).unsqueeze(2)
    preds_tail = torch.argmax(logit_tail, dim=2).unsqueeze(2)
    pred = torch.cat([preds_head,preds_tail],dim=2)

    # for i in range(ids.shape[0]):
    #     decoded.append(decode(ids[i],pred[i]))
    return ids,pred
def decode_diyawu(head,tail,maxlength=None):
    """
    head list : maxlength
    tail list: maxlength
    """
    res=[]
    i=0
    while i < maxlength:
        if head[i] == 1:
            find = False
            for j in range(maxlength):
                if j>i:
                    if tail[j] == 1:
                        find = True
                        res.append([i,j])
                        break
            if find:
                i = j+1
            else:
                break
        else:
            i+=1
    return res

def decode(ex,label):
    """
    ex: input id token [m,]
    label : [m,2]
    """
    head = label[:,0].tolist()
    tail = label[:,1].tolist()
    piece = decode_diyawu(head,tail,maxlength=len(head))
    def convert2text(ex,s,e):
        return ''.join(tokenizer.convert_ids_to_tokens(ex[s:e+1]))
    return [convert2text(ex,pi[0],pi[1]) for pi in piece]

def remove_dupdict(org):
    # org = [{'asd':"asdsad"},{'asd':"asdsad"},{"asda":{"asdaa":"asdsad"}},{"asda":{"asdaa":"asdsad"}}]
    res = []
    for orgi in org:
        exist = False
        for ri in res:
            if ri == orgi:
                exist = True
        if exist == False:
            res.append(orgi)
    return res

def exact_baozheng_relation(x):
    """
    str x : 裁判文书
    list jine_list : 提取的保证金额列表
    list[dict] y : 保证人-保证金额对
    """
    # panjue = exact_panjue(x)
    # # print(panjue)
    # parttext = [panjue[:gi.span()[1]] for gi in re.finditer('(连带)|(清偿责任)',panjue)]
    # # print(parttext)
    # # keytext = [x[[gi for gi in re.finditer('\n[一二三四五六七八九][、，]',x)][-1].span()[-1]:] for x in parttext]
    # # keytext = [x[[gi for gi in re.finditer('\n[一二三四五六七八九][、，]',x)][-1].span()[-1]:] for x in parttext if [gi for gi in re.finditer('\n[一二三四五六七八九][、，]',x)]]
    # # exact_baozheng_relation(x)
    # # parttext
    # keytext = []
    # for i in range(len(parttext)):
    #     g = re.finditer('(\n)*[一二三四五六七八九][、，]',parttext[i])
    #     list = [gi for gi in g]
    #     if list:
    #         key = parttext[i][list[-1].span()[-1]:]
    #         keytext.append(key)
        
    # jine = [exact_jine_all(ki)[0][0] if len(exact_jine_all(ki))!=0 else '' for ki in keytext]
    # baozhengren = [exact_baozhengren(ki) for ki in keytext]
    panjue = exact_panjue(x)
    parttext = [panjue[:gi.span()[1]] for gi in re.finditer('(连带清偿责任)',panjue)]
    # parttext
    keytext = []
    for i in range(len(parttext)):
        g = re.finditer('(\n)*[一二三四五六七八九][、，]',parttext[i])
        list = [gi for gi in g]
        if list:
            key = parttext[i][list[-1].span()[-1]:]
            keytext.append(key)
    # keytext
    jine = [exact_jine_all(ki)[0][0] if len(exact_jine_all(ki))!=0 else '' for ki in keytext]
    # jine
    baozhengren = [exact_baozhengren(ki)[:-1] for ki in keytext]
    res = []
    for i in range(len(jine)):
        resi = {}
        resi['maximum_guaranteed_amount'] = jine[i]
        resi['guarantor'] = baozhengren[i]
        res.append(resi)
    return remove_dupdict(res)

def exact_diyawu_rule(short_text):
    """
    short_text str: 短文本
    output list: 抵押物列表 
    """
    x = short_text
    lac_result = lac.run(x)
    res = []
    str=''
    for i in range(len(lac_result[0])):
        if lac_result[1][i] == 'LOC':
            str+=lac_result[0][i]
        else:
            if str != '':
                res.append(str)
                str=""
    return res

def exact_loc(x):
    lac_result = lac.run(x)
    res = []
    str=''
    for i in range(len(lac_result[0])):
        if lac_result[1][i] == 'LOC':
            str+=lac_result[0][i]
        else:
            if str != '':
                res.append(str)
                str=""
    return res

def exact_diyawu_omodel(short_text):
    x = short_text

    # 预处理
    id, mask = preprocessing_for_bert([x],MAX_OMODEL_LENGTH)

    # 制作loader
    batch_size = 1
    test_data_ = TensorDataset(id, mask)
    test_sampler = SequentialSampler(test_data_)
    test_dataloader = DataLoader(test_data_, sampler=test_sampler, batch_size=batch_size)

    # 传入模型预测金额类别
    ids,preds= omodel_predict(o_model,test_dataloader,label=False)
    id = ids[0]
    pred = preds[0]
    res = decode(id,pred)
    res = [resi for resi in res if '名下' not in resi and '有权以依法' not in resi]
    res = unpack_location(res)
    res = [xi.replace("#","") for xi in res]
    return res

def exact_diyawu_relation(x):
    """
    str x : 裁判文书
    list[dict] y : 抵押物-最高抵押金额
    """
    panjue = exact_panjue(x)
    # print(panjue)
    parttext = [panjue[:gi.span()[1]] for gi in re.finditer('(优先受偿)',panjue)]
    # parttext
    keytext = []
    for i in range(len(parttext)):
        g = re.finditer('(\n)*[一二三四五六七八九][、，]',parttext[i])
        list = [gi for gi in g]
        if list:
            key = parttext[i][list[-1].span()[-1]:]
            keytext.append(key)
    # keytext
    jine = [exact_jine_all(ki)[0][0] if len(exact_jine_all(ki))!=0 else '' for ki in keytext]
    diyawu = [exact_diyawu_omodel(ki) for ki in keytext]
    res = []
    for i in range(len(jine)):
        resi = {}
        resi['maximum_mortgage_amount'] = jine[i]
        resi['mortgage'] = diyawu[i]
        res.append(resi)
    return remove_dupdict(res)

def exact_panjue(x):
    # 关键词
    start_key = ['裁定如下','判决如下','裁判主文','特发出如下支付令','如下调解协议','本院分析如下','判决结果']
    end_key = ['审判长','审判员','如不服','受理费','书记员']
    
    # 开头
    start = 0
    for i in start_key:
        pos = x.find(i)
        if pos != -1:
            if i == '判决如下':
                start = pos
                break
            start = max(pos,start) # 最右边界准则
    # 结尾
    end = float('inf')
    for i in end_key:
        pos = x.find(i)
        if pos != -1 and pos >= start:
            end = min(pos,end) # 最左边界准则
    # 返回
    if end != float('inf') and start != 0 and end >= start:
        return x[start:end]
    else:
        # print(start,end)
        return ''


def text_preprocessing(sentence):
    # 去掉换行的
    sentence = sentence.replace('\n', '')
    def is_chinese(uchar):
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False

    def format_str(content):
        content_str = ''
        for i in content:
            if is_chinese(i):
                content_str = content_str + ｉ
        return content_str

    # 让文本只保留汉字
    sentence = format_str(sentence)

    # 分词
    sentence_seg = [i for i in jieba.cut(sentence)]

    # 去掉停词
    filter_sentence= [w for w in sentence_seg if w not in stop]

    return ' '.join(filter_sentence)

def predict_panjue_class(x):
    if type(x) == str:
        x = [x]
    X_preprocessed = np.array([text_preprocessing(text) for text in x])
    X_f = tf_idf.transform(X_preprocessed)
    probs = nb_model.predict_proba(X_f)

    return np.argmax(probs,axis=1)


def save_skmodel(model,path):
    joblib.dump(model, path)

def read_skmodel(path):
    return joblib.load(path)


def predict_skmodel(x,tfmodel,nbmodel):
    if type(x) == str:
        x = [x]
    X_preprocessed = np.array([text_preprocessing(text) for text in x])
    X_f = tfmodel.transform(X_preprocessed)
    probs = nbmodel.predict_proba(X_f)
    return np.argmax(probs,axis=1)


def predict_diyawu(x):
    return predict_skmodel(x,tfmodel,nbmodel)[0]
    

# 标签语义映射
LABEL_MAP = {"be":'principal_balance',
"he":'total_interest',"li":'interest',"fa":'payment_balance',"fu":"compound_interest_balance",
"ba":'maximum_guaranteed_amount',"di":'maximum_mortgage_amount',"qi":"other_amount",'bx':"sum_insterest_debt"}

LABEL_ENCODE = {"be":0,
"he":1,"li":2,"fa":3,"fu":4,
"ba":5,"di":6,"qi":7,'bx':8}

LABEL_DECODE = zip(LABEL_ENCODE.values(), LABEL_ENCODE.keys())
LABEL_DECODE = dict(LABEL_DECODE)


# 数据预处理函数
def text_preprocessing_bert(x):
    return x
# Create a function to tokenize a set of texts
def preprocessing_for_bert(data,MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing_bert(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            padding='max_length',                  # Max length to truncate/pad
            max_length=MAX_LEN,
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



# 预测
def bert_predict(model, test_dataloader,label = True):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
    #     # Load batch to GPU
# Load batch to GPU
        if label :
            leftid, leftmask, rightid, rightmask, _= tuple(t.to(device) for t in batch)
        else:
            leftid, leftmask, rightid, rightmask = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(leftid, leftmask, rightid, rightmask)
        # Compute logits
        # with torch.no_grad():
        #     logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, loss_fn=None,freeze_bert=False,bert_path=None):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in,D_out = 1 * 312 * 2,len(LABEL_ENCODE)


        # Instantiate BERT model
        self.bert  = AutoModel.from_pretrained(bert_path)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.loss = loss_fn

    def forward(self, input_ids_left, attention_mask_left,input_ids_right, attention_mask_right):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs_left = self.bert(input_ids=input_ids_left,
                            attention_mask=attention_mask_left)

        outputs_right = self.bert(input_ids=input_ids_right,
                            attention_mask=attention_mask_right)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        left_token = outputs_left[0][:, 0, :]
        right_token = outputs_right[0][:, 0, :]
        batchsize=left_token.size(0)
        last_hidden_state_cls_l = left_token.reshape(batchsize,-1)
        last_hidden_state_cls_r = right_token.reshape(batchsize,-1)
        cls_cat = torch.cat([last_hidden_state_cls_l,last_hidden_state_cls_r],dim=1)
        # Feed input to classifier to compute logits
        logits = self.classifier(cls_cat)

        return logits
    

def exact_nine_jine_model(x):
    # 判决部分的提取
    panjue = exact_panjue(x)

    if panjue:
        # 初步判定是否含有金额实体
        exist = predict_panjue_class(panjue)[0]

        if exist:

            # 提取金额实体对
            # 收集所有的金额实体
            jine = exact_jine_all(panjue)
            if len(jine) == 0:
                return None
            hashmap = {}
            for i,ji in enumerate(jine):
                label = 'unknow'
                hashmap[ji] = label
            
            # 提取金额附近的文本
            k=30
            test_lefttexts = []
            test_righttexts = []
            for kii in hashmap.keys():
                lefttext = panjue[max(0,kii[1][0]-k):kii[1][0]]
                righttext = panjue[kii[1][1]:min(len(panjue),kii[1][1]+k)]
                test_lefttexts.append(lefttext)
                test_righttexts.append(righttext)

            # 预处理
            batch_size = 1
            test_inputs_left, test_masks_left = preprocessing_for_bert(test_lefttexts,MAX_JINEMODEL_LENGTH)
            test_inputs_right, test_masks_right = preprocessing_for_bert(test_righttexts,MAX_JINEMODEL_LENGTH)

            # 制作loader
            test_data_ = TensorDataset(test_inputs_left, test_masks_left,test_inputs_right, test_masks_right)
            test_sampler = SequentialSampler(test_data_)
            test_dataloader = DataLoader(test_data_, sampler=test_sampler, batch_size=batch_size)

            # 传入模型预测金额类别
            y_pred_all = bert_predict(jine_model,test_dataloader,label=False)
            y_pred_all_cate = np.argmax(y_pred_all,1)

            # 整理格式输出

            # 格式处理
            k=0
            for ki,vi in hashmap.items():
                hashmap[ki] = LABEL_MAP[LABEL_DECODE[int(y_pred_all_cate[k])]]
                k+=1

            output = {}
            for i in LABEL_MAP.values():
                output[i] = []

            for ki,vi in (hashmap.items()):
                output[vi].append(ki[0])

            return output
def get_jine_model(jinemodelpath):
    loss_fn = nn.CrossEntropyLoss(weight=torch.rand(len(LABEL_ENCODE),))
    model = BertClassifier(freeze_bert=True,loss_fn=loss_fn,bert_path=modelpath
+'bert').to(device)
    model.load_state_dict(torch.load(jinemodelpath,map_location=torch.device('cpu')))
    return model

def get_diyawu_model(diyawupath):
    loss_fn = nn.CrossEntropyLoss(weight=torch.rand(2,))
    model = Omodel(freeze_bert=True,loss_fn=loss_fn,bert_path=modelpath
+'bert').to(device)
    model.load_state_dict(torch.load(diyawupath,map_location=torch.device('cpu')))
    return model



# 读停词
def stopwordslist():
    stopwords = [line.strip() for line in open(datapath+'hit.txt',encoding='UTF-8').readlines()]
    return stopwords

empty_template1={
        "guarantor-maximum_guaranteed_amount": [
        ],
        "deadline": [
        ],
        "mortgage-maximum_mortgage_amount": [
        ],
        "amount": {
            "maximum_guaranteed_amount": [
                
            ],
            "other_amount": [],
            "interest": [],
            "total_interest": [
                
            ],
            "compound_interest_balance": [],
            "maximum_mortgage_amount": [
                
            ],
            "sum_insterest_debt": [],
            "principal_balance": [
                
            ],
            "payment_balance": [],
            "debt_amount": [
             
            ]
        }
    }


empty_template2={
        "mortgage-maximum_mortgage_amount": []
}

def exact_jine_all(x):
    """金额字段的正则提取器"""
    res = get_jine_g(x)
    g1 = res['数字']
    g2 = res['中文']
    res1 = [((float(gi.group(1).replace(",","").replace("，","")),gi.group(4)),gi.span()) for gi in g1]
    res2 = [((float(cn2an.cn2an(gi.group(1)+gi.group(2), "smart")),gi.group(3)),gi.span()) for gi in g2]
    return res1+res2

def get_jine_g(x):
    # 数字金额
    Regx = re.compile("(([1-9]\\d*[\\d,，]*\\.?\\d*)|(0\\.[0-9]+))(元|百万|万元|亿元|万|亿|人民币|美元|美金)")
    res = re.finditer(Regx,x)

    # 中文金额
    p=r'(一|二|三|四|五|六|七|八|九|十)+([一|二|三|四|五|六|七|八|九|十|百|千|万|亿]+)(元|美金|人民币)'
    res2 = re.finditer(p,x)
    
    return {"数字":res,"中文":res2}

def unpack_location(inp):
    # inp = ['sss，xxx','sss','sss,jjj']
    res = []
    for i in inp:
        temp = []
        for j in i.split('，'):
            for k in j.split(','):
                temp.append(k)
        res+=temp
    return res


datapath = r'./data/'
modelpath = r'./model/'

# 停用词
stop = stopwordslist()

# 装载LAC模型
lac = LAC(mode='lac')

# 金额文本分类模型
tf_idf = joblib.load(modelpath+r'panjue_exist/tf_fenlei.pkl')
nb_model = joblib.load(modelpath+r'panjue_exist/fenlei.pkl')

tfidf_path = modelpath+r'diyawu_exist/tf_idf.pkl'
nb_path = modelpath+r'diyawu_exist/nb_model.pkl'
tfmodel = read_skmodel(tfidf_path)
nbmodel = read_skmodel(nb_path)

tokenizer_path = modelpath+"bert"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

omodel_path = modelpath+'omodel_v4_max306.ckpt'
o_model = get_diyawu_model(omodel_path)

jine_model_path = modelpath+'001_wacc_94.pt'
jine_model = get_jine_model(jine_model_path)



empty_template1={
        "guarantor_maximum_guaranteed_amount": [
        ],
        "deadline": [
        ],
        "mortgage_maximum_mortgage_amount": [
        ],
        "amount": {
            "maximum_guaranteed_amount": [
                
            ],
            "other_amount": [],
            "interest": [],
            "total_interest": [
                
            ],
            "compound_interest_balance": [],
            "maximum_mortgage_amount": [
                
            ],
            "sum_insterest_debt": [],
            "principal_balance": [
                
            ],
            "payment_balance": [],
            "debt_amount": [
             
            ]
        }
    }


empty_template2={
        "mortgage_maximum_mortgage_amount": [
        ]
}



from flask import Flask, request, jsonify
import json
 
app = Flask(__name__)
app.debug = False

@app.route('/')
def index():
    return '金额提取模块,对/extract_money/发送POST请求进行测试。'


@app.route('/extract_money/',methods=['post'])
def exact_jine():
    out = {}
    out['msg'] = 0
    out['content'] = empty_template1
    try:
        input = request.data.decode('utf-8')
        input_dict = json.loads(input)
        if not request.data or 'text' not in input_dict:   #检测是否有数据
            return (jsonify(out))
        text = input_dict['text']
        out['msg'] = 1
        out['content'] = exact_all(text)
        return jsonify(out)
    except:
        out['msg'] = 0
        return jsonify(out)
        
@app.route('/extract_mortgage/',methods=['post'])
def exact_mortgage():
    out = {}
    out['msg'] = 0
    out['content'] = empty_template2
    try:
        input = request.data.decode('utf-8')
        input_dict = json.loads(input)
        if not request.data or 'text' not in input_dict:   #检测是否有数据
            return (jsonify(out))
        text = input_dict['text']
        out['msg'] = 1
        out['content'] = exact_mortgage_(text)
        return jsonify(out)
    except:
        out['msg'] = 0
        return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=18080)

