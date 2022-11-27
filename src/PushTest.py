
from transformers import BertForMaskedLM,BertTokenizer
import torch
import numpy as np
import pandas as pd
from numpy.random import choice
import pickle
from tqdm import tqdm
import json
import glob
from pathlib import Path
from datetime import datetime
import argparse
from transformers import logging
logging.set_verbosity_error()

def ScoreLabels(vocab: list, scores: list) -> list:
    """Generates list of (vocab,score) tuples sorted by score in descending order.

    Args:
        vocab ([List]): PLM vocabulary
        scores ([List]): Score of each token

    Returns:
        [List]: List of (vocab,score) tuples
    """
    # TODO: Can be enhanced using ordered tuples?
    return sorted(dict(zip(vocab, scores)).items(), key=lambda x: -x[1])
def pred(sent,model):
    input_ids = tokenizer.encode_plus(sent,return_tensors='pt')['input_ids']
    mask_id = [ind for ind,x in enumerate(input_ids[0]) if x == tokenizer.mask_token_id][0]
    with torch.no_grad():
        o = model(input_ids)
    s=dict(ScoreLabels(vocab,o.logits[0][mask_id]))
    s={k:v.detach().cpu().item() for (k,v) in s.items()}
    return s


def normalize(s):
    vals = list(s.values())
    n_vals = np.exp(vals)/np.sum(np.exp(vals))
    n_s = dict(zip(list(s.keys()),n_vals))
    return n_s

my_parser = argparse.ArgumentParser(description='Push Test')




my_parser.add_argument('--k',
                       type=int,
                       help='TE multiplier')

my_parser.add_argument('--model_arch',
                       type=str,
                       default='bert-base-cased',
                       help='model used')

my_parser.add_argument('--num_runs',
                       type=int,
                       default=10,
                       help='number of runs')

my_parser.add_argument('--num_samples',
                       type=int,
                       default=100,
                       help='number of samples')

my_parser.add_argument('--type',
                       type=str,
                       help='Type')

my_parser.add_argument('--seed',
                       type=int,
                       default=0,
                       help='Seed for reproducibility')


args = my_parser.parse_args()



k=args.k#1
model_arch =args.model_arch# 'bert-base-cased'
num_runs =args.num_runs#10
num_samples = args.num_samples #100
type_ = args.type# 'City'
seed = args.seed #0

print(type_)


model = BertForMaskedLM.from_pretrained(model_arch)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_arch)
vocab = list(tokenizer.get_vocab().keys())







pr='[MASK]'
with open(f'TypeVectors_10/{type_}_vectors.pkl','rb') as f:
    d = pickle.load(f)
type_vector = d['svd_vec']
vec_samples = d['samples']

type_files = glob.glob(f'ProcessedDatasets/synth_{type_}/**/*')
type_prompt_df = pd.concat([pd.DataFrame(pd.read_csv(x)) for x in type_files])
type_prompt_df=type_prompt_df.dropna()
all_type_prompts = list(type_prompt_df.sent)

type_df = pd.read_csv(f'KG Samples/{type_}.csv')
type_df['Label'] = type_df.Label.apply(lambda x: str(x))
type_df = type_df[~type_df.Label.apply(lambda x: True if x in vec_samples else False)]
type_df=type_df[type_df.Label != 'nan']
type_df=type_df[type_df.Label.apply(lambda x: True if x==x else False)]
kg_samples = list(type_df.Label)



importances = np.array(type_df.Degree)
importances_p=importances/sum(importances)


pos_types = [x for x in kg_samples]
neg_types = [x for x in vocab if x not in kg_samples and '#' not in x]




np.random.seed(seed)
seeds =  choice(range(1000),size =num_runs,replace = False)



pos_push_wo_arr = []
pos_push_rand_arr = []

res = {}
output_path = f"PushTests/{type_}/"
Path(output_path+f"{k}/{num_samples}/{seed}/").mkdir(parents=True, exist_ok=True)


for i in tqdm(range(num_runs)):

    np.random.seed(seeds[i])
    sampled_prompts = choice(all_type_prompts,size = num_samples)

    np.random.seed(seeds[i])
    sampled_pos_types = choice(pos_types,size = min(len(pos_types),num_samples),replace = True,p = importances_p)

    np.random.seed(seeds[i])
    sampled_neg_types = choice(neg_types,size = num_samples)


    # without TE
    wo_TE = pred(pr,model)
    n_wo_TE = normalize(wo_TE)
    # with TE: rank(pos) > rank(neg)
    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]+=k*type_vector

    w_TE = pred(pr,model)    
    n_w_TE = normalize(w_TE)

    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]+=k*(-type_vector)


    #compute
    pos_push_wo = 0
    false_TE=[]
    for p in sampled_pos_types:


        rank_before_TE = list(n_wo_TE.keys()).index(p)
        rank_after_TE = list(n_w_TE.keys()).index(p)

        score_before_TE = n_wo_TE[p]
        score_after_TE = n_w_TE[p]




        #if not(rank_after_TE>rank_before_TE and score_after_TE<score_before_TE): pos_push_wo+=1
        if score_after_TE>score_before_TE: pos_push_wo+=1
        else: 
            false_TE.append([(p,rank_after_TE,score_after_TE)])
            #print(p)


    pos_push_wo_arr.append(pos_push_wo/len(sampled_pos_types))

res['pos_push_wo_arr']=pos_push_wo_arr

res['mean_pos_push_wo_arr']=np.mean(pos_push_wo_arr)


res['std_pos_push_wo_arr']=np.std(pos_push_wo_arr)

res['wo_TE'] = wo_TE
res['w_TE'] = w_TE
res['false_TE']=false_TE

print('mean: ',np.mean(pos_push_wo_arr))
print('std: ', np.std(pos_push_wo_arr))


with open(output_path+f"{k}/{num_samples}/{seed}/{datetime.now().strftime('%Y%m-%d%H-%M%S')}_res.json",'w') as f:
    json.dump(res,f,indent=4)

