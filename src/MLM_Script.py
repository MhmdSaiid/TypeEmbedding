import json
import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer,AutoModelForMaskedLM,AutoConfig,BertTokenizer,BertForMaskedLM
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
import os
import argparse
from transformers import logging
logging.set_verbosity_error()

my_parser = argparse.ArgumentParser(description='MLM Model')



my_parser.add_argument('--file',
                       type=str,
                       help='path to data')

my_parser.add_argument('--model_arch',
                       type=str,
                       default='bert-base-cased',
                       help='model used')

my_parser.add_argument('--concept_vector',
                       type=str,
                       default=None,
                       help='concept vector')

my_parser.add_argument('--k',
                       type=int,
                       default=0,
                       help='concept vector multiplier')

my_parser.add_argument('--mean',
                       action='store_true',
                       default=False,
                       help='Retrieves the mean vector, otherwise SVD')

my_parser.add_argument('--seed',
                       type=int,
                       default=0,
                       help='Seed for reproducibility')


my_parser.add_argument('--holdout_ratio',
                       type=float,
                       default=0.05,
                       help='holdout ratio')

my_parser.add_argument('--manual_k',
                       action='store_true',
                       default=False,
                       help='Manual setting of k')

my_parser.add_argument('--token_baseline',
                       action='store_true',
                       default=False,
                       help='do token baseline')

my_parser.add_argument('--method_label',
                       type=str,
                       help='Method Label')


args = my_parser.parse_args()



k=args.k #1 #vector multiplier
model_arch = args.model_arch #'bert-base-cased'
concept_vector_file = args.concept_vector #torch.rand(768)
file = args.file #'P1290.jsonl'
isMean = args.mean
seed = args.seed
concept_vector=None
holdout_ratio = args.holdout_ratio
manual_k = args.manual_k
token_baseline = args.token_baseline
method_label = args.method_label


comp_type = 'mean' if isMean else 'svd'
optimal_k = float('inf')

if concept_vector_file:
  with open(concept_vector_file,'rb') as f:
    concept_vector_d=pickle.load(f)
concept_vector = concept_vector_d['svd_vec'] if not isMean else concept_vector_d['mean_vec']
vec_samples = concept_vector_d['samples']

def load_jsonl(file):
    data=[]
    with open(file,'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def ScoreLabels(vocab: list, scores: list) -> list:
    # TODO: Can be enhanced using ordered tuples?
    return sorted(dict(zip(vocab, scores)).items(), key=lambda x: -x[1])


def preprocess(data):
    data_df=[]
    for i in tqdm(range(len(data))):
        x = data.iloc[i]
        encoded = tokenizer.encode_plus(x.sent, return_tensors='pt')#['input_ids']
        mind = (encoded['input_ids'][0]==tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if list(mind):
            data_df.append([x.sent,encoded['input_ids'],mind,x.gold])

    data_df = pd.DataFrame(data_df,columns=['data','encoded_data','mask_ind','gold'])
    data_df.gold=data_df.gold.apply(lambda x: str(x))
    data_df['gold_ind'] = data_df.gold.apply(lambda x: vocab.index(x))
    data_df['len']=data_df.apply(lambda x: len(x.encoded_data[0]),axis=1)
    data_df=data_df[data_df.len<=model.config.max_position_embeddings]
    data_df_len = pd.DataFrame(data_df.groupby('len')['data'].apply(list))
    data_df_len['encoded_data']=data_df.groupby('len')['encoded_data'].apply(list)
    data_df_len['mask_ind']=data_df.groupby('len')['mask_ind'].apply(list)
    data_df_len['gold']=data_df.groupby('len')['gold'].apply(list)
    data_df_len['gold_ind']=data_df.groupby('len')['gold_ind'].apply(list)
    return data_df_len



def infer(data_df_len,k):
    ans=[]
    p1=[]
    p10=[]
    p50=[]
    p100=[]

    for l in tqdm(data_df_len.index):

        b_data=data_df_len.loc[l].data
        b_encoded_data=data_df_len.loc[l].encoded_data
        b_mask_ind=data_df_len.loc[l].mask_ind
        b_gold_ind=data_df_len.loc[l].gold_ind

        test_input_ids = torch.cat(b_encoded_data, dim=0).to(device)
        test_mask_inds = torch.cat(b_mask_ind, dim=0).to(device)
        test_gold_ind = torch.tensor(b_gold_ind).to(device)

        b_ans=[]

        test_dataset = TensorDataset(test_input_ids,test_mask_inds,test_gold_ind)
        test_dataloader = DataLoader(dataset=test_dataset,
                                sampler=SequentialSampler(test_dataset),
                                batch_size=32,
                                )
        for batch in test_dataloader:
            bb_encoded_data = batch[0].to(device)
            bb_minds = batch[1].to(device)
            bb_gold_ind = batch[2]
            
            with torch.no_grad():
                o = model(bb_encoded_data)
                bb_ans = [vocab[torch.argmax(x[ind,:])] for ind,x in zip(bb_minds,o.logits)]
                top100s = [reversed(torch.argsort(x[ind])[-101:]) for ind,x in zip(bb_minds,o.logits)]
                #print([x[0] for x in top100s])
                top100s = [x[1:] if x[0].item() == tokenizer.mask_token_id else x[:100] for x in top100s]
                #print([x[0] for x in top100s])
                #print("\n")
                p1_ans = [ 1 if ind in x[:1] else 0     for ind,x in zip(bb_gold_ind,top100s)]
                p10_ans = [ 1 if ind in x[:10] else 0     for ind,x in zip(bb_gold_ind,top100s)]
                p50_ans = [ 1 if ind in x[:50] else 0     for ind,x in zip(bb_gold_ind,top100s)]
                p100_ans = [ 1 if ind in x[:100] else 0     for ind,x in zip(bb_gold_ind,top100s)]

                b_ans.extend(bb_ans)
                p1.extend(p1_ans)
                p10.extend(p10_ans)
                p50.extend(p50_ans)
                p100.extend(p100_ans)

        ans.append(b_ans)

    data_df_len['ans']=ans


    p1_acc = sum(p1)/len(p1)
    p10_acc = sum(p10)/len(p10)
    p50_acc = sum(p50)/len(p50)
    p100_ac = sum(p100)/len(p100)

    print(f'P@1:{p1_acc}')
    print(f'P@10:{p10_acc}')
    print(f'P@50:{p50_acc}')
    print(f'P@100:{p100_ac}')

    res={
        'model':model_arch,
        'k':k,
        'concept_vector':concept_vector_file,
        'vectorType': comp_type,
        'file':file,
        'P@1':p1_acc,
        'P@10':p10_acc,
        'P@50':p50_acc,
        'P@100':p100_ac
    }

    return res



#load model
#config = AutoConfig.from_pretrained(model_arch)
tokenizer = BertTokenizer.from_pretrained(model_arch)
#model = AutoModelForMaskedLM.from_config(config)
model = BertForMaskedLM.from_pretrained(model_arch)

model.eval()
vocab = list(tokenizer.get_vocab().keys())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
concept_vector=concept_vector.to(device)

#prepare data
data = pd.read_csv(file).iloc[:]

data,holdout_data = train_test_split(data, test_size=holdout_ratio, random_state=seed)
rel = file.split('/')[-1].replace('.csv','')
tmp_file = f'{rel}_holdout.csv'



#find_optimal k
if not manual_k and not token_baseline:
    sh1=time.time()
    holdout_data_len=preprocess(holdout_data)
    eh1=time.time()

    for l in range(0,6):
        print("\n\n")
        #update mask embd
        if concept_vector is not None and l>0:
            with torch.no_grad():
                model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]+=l*concept_vector

        print(f'k={l}')
        sh2=time.time()
        res = infer(holdout_data_len,l)
        eh2=time.time()


        res['process_time']=eh1-sh1
        res['infer_time']=eh2-sh2

        print(f'Processing Time:{eh1-sh1}')
        print(f'Infer Time:{eh2-sh2}')

        folder_path = f'results/holdout/{file}/{model_arch}/{method_label}/{concept_vector_file.split("/")[0]}/{l}'

        Path(folder_path).mkdir(parents=True, exist_ok=True)
        json.dump(res, open(f"{folder_path}/res.json", "w"), indent=4)



        #restire mask embd
        if concept_vector is not None and l>0:
            with torch.no_grad():
                model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]-=l*concept_vector



    tmp_folder_path = f'results/holdout/{file}/{model_arch}/{method_label}/{concept_vector_file.split("/")[0]}/'

    d={}

    for l in os.listdir(tmp_folder_path):
        for _file in os.listdir(tmp_folder_path+str(l)):
            with open(tmp_folder_path+str(l)+'/'+_file,'r') as f:
                res_l = json.load(f)
                d[l] = res_l

    tmp_res_df = pd.DataFrame.from_dict(d).T[['k','P@1','P@10','P@50','P@100']]
    tmp_res_df=tmp_res_df.set_index('k')
    tmp_res_change_df=tmp_res_df.subtract(tmp_res_df.loc[0])

    max_val = tmp_res_change_df.max().max()

    for ind in tmp_res_change_df.index:
        if max_val in set(tmp_res_change_df.loc[ind]):
            if ind < optimal_k:
                optimal_k = ind

    print(f'\n\nOptimal k: {optimal_k}')




data=data.iloc[:]
s1=time.time()

if token_baseline:
    #joined_samples = " ".join(vec_samples)
    #data.sent = data.sent.apply(lambda x: x.replace('[MASK]',f'{joined_samples} [MASK]'))
    type_ = concept_vector_file.split('/')[1].replace('_vectors.pkl','').replace('_',' ').lower()
    data.sent = data.sent.apply(lambda x: x.replace('[MASK]',f'the {type_} [MASK]'))


data_df_len = preprocess(data)
e1=time.time()
if not manual_k: k = optimal_k

if not token_baseline and concept_vector is not None and k>0:
    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]+=k*concept_vector

s2=time.time()
res =infer(data_df_len,k)
e2=time.time()

res['process_time']=e1-s1
res['infer_time']=e2-s2
res['method_label'] = method_label

print(f'Processing Time:{e1-s1}')
print(f'Infer Time:{e2-s2}\n')

#folder_path = f'results/{file}/{model_arch}/{comp_type}/{concept_vector_file.split("/")[0]}/{k}'
folder_path = f'results/{file}/{model_arch}/{method_label}/{concept_vector_file.split("/")[0]}'

Path(folder_path).mkdir(parents=True, exist_ok=True)
json.dump(res, open(f"{folder_path}/res.json", "w"), indent=4)
