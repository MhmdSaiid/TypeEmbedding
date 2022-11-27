import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import choice
from transformers import AutoTokenizer, AutoModelForMaskedLM




# Functions to get Type Vectors
def get_average_vec(WE_module,vocab,sampled_tokens):
    """_summary_

    Args:
        WE_module (_type_): PLM Word Embedding Module
        vocab (_type_): Vocab of PLM
        concept_tokens (_type_): List of tokens represenitng concept
        sample_size (_type_): sample size
        seed (_type_): seed value

    Returns:
        _type_: concept vector through average
    """    
    token_indices=[vocab.index(n) for n in sampled_tokens]
    with torch.no_grad():
        mean_type_vector = torch.mean(torch.stack([WE_module(torch.tensor(i)) for i in token_indices]),dim=0)
    return mean_type_vector
    

def get_svd_vec(WE_module,vocab,sampled_tokens):
    """_summary_

    Args:
        WE_module (_type_): PLM Word Embedding Module
        vocab (_type_): Vocab of PLM
        concept_tokens (_type_): List of tokens represenitng concept
        sample_size (_type_): sample size
        seed (_type_): seed value

    Returns:
        _type_: concept vector through svc
    """     
    
    token_indices=[vocab.index(n) for n in sampled_tokens]
    with torch.no_grad():
        stacked_vectors = torch.stack([WE_module(torch.tensor(i)) for i in token_indices]).detach().numpy()
    u, s, vt = np.linalg.svd(stacked_vectors, full_matrices=True)
    svd_vector = torch.tensor(-vt[0])
    return svd_vector


def vocab_sim(type_embedding,WE_module,vocab):
    '''similairty with PLM vocabulary '''
    WE_matrix = WE_module.weight.detach()
    sim_scores = torch.nn.functional.cosine_similarity(WE_matrix,type_embedding.reshape(1,-1)).numpy().tolist()
    d = dict(zip(vocab,list(sim_scores)))
    sorted_scores = sorted(d.items(),key=lambda x: -x[1])
    return sorted_scores






my_parser = argparse.ArgumentParser(description='Generate TE')



my_parser.add_argument('--model_arch',
                       type=str,
                       default='bert-base-cased',
                       help='model used')

my_parser.add_argument('--path',
                       type=str,
                       default='data/KG Samples/',
                       help='path to samples')

my_parser.add_argument('--seed',
                       type=int,
                       default=0,
                       help='Seed for reproducibility')

my_parser.add_argument('--num_samples',
                       type=int,
                       default=10,
                       help='Number of Samples')

my_parser.add_argument('--sample_type',
                       type=str,
                       default='Weighted',
                       choices=['Weighted','Top','Bot','Unif'],
                       help='Sampling Method')


args = my_parser.parse_args()


model_arch = args.model_arch
path = args.path
seed = args.seed
num_samples = args.num_samples
sample_type = args.sample_type

output_path = f'data/TypeVectors_{num_samples}_{model_arch}_{sample_type}/'


model = AutoModelForMaskedLM.from_pretrained(model_arch)
device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_arch)
vocab = list(tokenizer.get_vocab().keys())


if __name__=='__main__':
    kg_samples = os.listdir(path)
    for x in kg_samples:
        
        type_ = x.replace('.csv','')
        print(f'Generating Type Vectors for {type_}')

        type_df = pd.read_csv(path+x)
        type_df.Label = type_df.Label.apply(lambda x: str(x))
        all_samples = list(type_df.Label)

        importances = np.array(type_df.Degree)
        importances_p=importances/sum(importances)

        if sample_type == 'Weighted':
            np.random.seed(seed)
            samples = choice(all_samples,size = min(num_samples,len(type_df)),replace = False,p = importances_p)        

        elif sample_type == 'Top':
            samples = list(type_df.sort_values(by='Degree',ascending=False)['Label'].iloc[:10])

        elif sample_type=='Bot':
            samples = list(type_df.sort_values(by='Degree',ascending=True)['Label'].iloc[:10])
        elif sample_type=='Unif':
            np.random.seed(seed)
            samples = choice(all_samples,size = min(num_samples,len(type_df)),replace = False)        

        print(samples)
        mean_vec = get_average_vec(model.bert.embeddings.word_embeddings,vocab,samples)
        svd_vec = get_svd_vec(model.bert.embeddings.word_embeddings,vocab,samples)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        d= {'samples':samples,
            'mean_vec': mean_vec,
            'svd_vec':svd_vec}

        with open(output_path + f'{"_".join(type_.split(" "))}_vectors.pkl','wb') as f:
            pickle.dump(d,f)
