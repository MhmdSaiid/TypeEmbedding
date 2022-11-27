import pandas as pd
import numpy as np
import os
import json
import argparse


my_parser = argparse.ArgumentParser(description='Print Average Results')

my_parser.add_argument('--res_dir',
                       type=str,
                       help='path to results')

my_parser.add_argument('--avg_avg',
                       action='store_true',
                       default=False,
                       help='Only Show Average of Average')


args = my_parser.parse_args()
path = args.res_dir
avg_avg = args.avg_avg

if path[-1]!='/': path+='/'

def get_mean_df(res_d):
    final_mean_d={}
    for key in res_d:
        all_res_d = res_d[key]
        mean_res_d = {k:round(np.mean(v),3) for (k,v) in all_res_d.items()}
        final_mean_d[key] = mean_res_d
    return pd.DataFrame(final_mean_d)



#path = "results/ProcessedDatasets/"
#avg=True
#avg_avg=True


mean_res={}

for concept_vector in os.listdir(path):
    for dataset_name_pre in os.listdir(path+concept_vector+'/'):
        
        mean_res[concept_vector+'_'+dataset_name_pre]={}
        dataset_type_res={}

        for dataset_name_suf in  os.listdir(path+concept_vector+'/'+dataset_name_pre+'/'):
            for model_arch in os.listdir(path+concept_vector+'/'+dataset_name_pre+'/'+dataset_name_suf):
    
                filename = concept_vector+'/'+dataset_name_pre+'/'+dataset_name_suf+'/'
                filepath = path+filename

                processed_file = dataset_name_pre+'/'+dataset_name_suf

                # get result dictionary
                res_d={}
                for dirpath, dirnames, filenames in os.walk(filepath):
                    for jsonfilename in filenames:
                        json_file = json.load(open(dirpath+'/'+jsonfilename,"r"))
                        res_k={}
                        for ind in json_file:
                            if 'P@' in ind: res_k[ind]=json_file[ind]
                        res_d[json_file['method_label']] = res_k
                
                for precision_key in ['P@1','P@10','P@50','P@100']:
                    for m in res_d:
                        if m not in dataset_type_res: dataset_type_res[m]={}
                        if precision_key not in dataset_type_res[m]: dataset_type_res[m][precision_key]=[]
                        dataset_type_res[m][precision_key].append(res_d[m][precision_key])


        mean_res[concept_vector+'_'+dataset_name_pre] = dataset_type_res






mean_mean = []
for key in mean_res:
    df = get_mean_df(mean_res[key])
    if not avg_avg:
        print(key)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.T[['P@1', 'P@10', 'P@50', 'P@100']])
        print('\n\n')
    mean_mean.append(df.T[['P@1','P@10','P@50','P@100']])
mean_mean=pd.concat(mean_mean)

mean_mean['index'] = list(mean_mean.index)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(mean_mean.groupby('index').mean().applymap(lambda x: round(x,3)))