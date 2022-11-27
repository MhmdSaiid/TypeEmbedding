from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from transformers import AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
from numpy.random import choice
import pickle
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from datetime import datetime
import json
import argparse
from transformers import logging
from pathlib import Path

logging.set_verbosity_error()

def preprocess(data):
    data_df=[]
    for i in range(len(data)):
        x = data.iloc[i]
        encoded = tokenizer.encode_plus(x.sent, return_tensors='pt')#['input_ids']
        mind = (encoded['input_ids'][0]==tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if list(mind):
            data_df.append([x.sent,encoded['input_ids'],mind,x.gold,x.y])

    data_df = pd.DataFrame(data_df,columns=['data','encoded_data','mask_ind','gold','y'])
    
    data_df.gold=data_df.gold.apply(lambda x: str(x))
    data_df['gold_ind'] = data_df.gold.apply(lambda x: vocab.index(x))
    data_df['len']=data_df.apply(lambda x: len(x.encoded_data[0]),axis=1)
    data_df=data_df[data_df.len<=model.config.max_position_embeddings]
    data_df_len = pd.DataFrame(data_df.groupby('len')['data'].apply(list))
    data_df_len['encoded_data']=data_df.groupby('len')['encoded_data'].apply(list)
    data_df_len['mask_ind']=data_df.groupby('len')['mask_ind'].apply(list)
    data_df_len['gold']=data_df.groupby('len')['gold'].apply(list)
    data_df_len['gold_ind']=data_df.groupby('len')['gold_ind'].apply(list)
    data_df_len['y']=data_df.groupby('len')['y'].apply(list)

    return data_df_len

def infer(data_df_len,layer):
    data_df=[]

    ans=[]
    embds=[]

    for l in data_df_len.index:

        b_data=data_df_len.loc[l].data
        b_encoded_data=data_df_len.loc[l].encoded_data
        b_mask_ind=data_df_len.loc[l].mask_ind

        test_input_ids = torch.cat(b_encoded_data, dim=0).to(device)
        test_mask_inds = torch.cat(b_mask_ind, dim=0).to(device)

        b_ans=[]
        vectors =[]
        test_dataset = TensorDataset(test_input_ids,test_mask_inds)
        test_dataloader = DataLoader(dataset=test_dataset,
                                sampler=SequentialSampler(test_dataset),
                                batch_size=32,
                                )
        for batch in test_dataloader:
            bb_encoded_data = batch[0].to(device)
            bb_minds = batch[1].to(device)
            
            with torch.no_grad():
                o = model(bb_encoded_data)
                bb_ans = [vocab[torch.argmax(x[ind,:])] for ind,x in zip(bb_minds,o.logits)]
                b_ans.extend(bb_ans)
                b_vectors=[x[m.item()].cpu().tolist() for x,m in zip(o.hidden_states[layer],bb_minds)]
                vectors.extend(b_vectors)


        ans.append(b_ans)
        embds.append(vectors)

    data_df_len['ans']=ans
    data_df_len['Xembed']=embds#.tolist()
    final_train_df = pd.DataFrame(data_df_len.apply(lambda x:list(zip(x.data,x.ans,x.Xembed,x.y)),axis=1).explode().tolist(),columns=['data','ans','Xembed','y'])
    return final_train_df
def flatten(x): return [xxx for xx in x for xxx in xx]


my_parser = argparse.ArgumentParser(description='Layerwise Test')


my_parser.add_argument('--folder',
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

my_parser.add_argument('--type',
                       type=str,
                       help='Type')

my_parser.add_argument('--max_k',
                       type=int,
                       help='max k')

my_parser.add_argument('--train_sample_size',
                       type=int,
                       default=500,
                       help='train samples size')

my_parser.add_argument('--num_trials',
                       type=int,
                       default=10,
                       help='number of trials ')


my_parser.add_argument('--seed',
                       type=int,
                       default=0,
                       help='Seed for reproducibility')


args = my_parser.parse_args()

folder = args.folder
concept_vector = args.concept_vector
model_name = args.model_arch
type_ = args.type # 'City'
max_k = args.max_k # 5
train_sample_size = args.train_sample_size # 100
num_trials = args.num_trials #=10
seed = args.seed #0


#model_name = 'bert-base-cased'
model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = [x[0] for x in sorted(tokenizer.get_vocab().items(),key=lambda x: x[1])]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)


with open(f'{concept_vector}/{type_}_vectors.pkl','rb') as f:
  d = pickle.load(f)
svd_vec = d['svd_vec'].to(device)



pos_data_files = glob.glob(f'{folder}/synth_{type_}/**/**')

neg_type_folders = [x for x in glob.glob(f'{folder}/**') if type_ not in x]
neg_data_files = [flatten([glob.glob(x+'/**/**') for x in neg_type_folders])][0]


np.random.seed(seed)
seeds =  choice(range(1000),size =num_trials,replace = False)


res={}

for k in range(max_k+1):
    print(f'k={k}')
    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]+=svd_vec

    res[k]={}

    for class_train_num in tqdm([train_sample_size]):#[10,30,50,100,500]):
        res[k][class_train_num]={}

        for i in range(0,13):
            res[k][class_train_num][i]={}

        best_cls=[]
        max_f1=float('-inf')
        

        for i in range(num_trials):

            print(f'Trial #{i}')
            pos_train_files, pos_test_files = train_test_split(pos_data_files,train_size=0.7,random_state = seeds[i])
            neg_train_files, neg_test_files = train_test_split(neg_data_files,train_size=0.7,random_state = seeds[i])

            pos_train_data = pd.concat([pd.read_csv(x) for x in  pos_train_files])
            pos_train_data = pos_train_data.drop_duplicates(subset='sent')

            pos_test_data = pd.concat([pd.read_csv(x) for x in  pos_test_files])
            pos_test_data = pos_test_data.drop_duplicates(subset='sent')

            neg_train_data = pd.concat([pd.read_csv(x) for x in  neg_train_files])
            neg_train_data = neg_train_data.drop_duplicates(subset='sent')

            neg_test_data = pd.concat([pd.read_csv(x) for x in  neg_test_files])
            neg_test_data = neg_test_data.drop_duplicates(subset='sent')


            pos_train_data['y']=1
            pos_test_data['y']=1
            neg_train_data['y']=0
            neg_test_data['y']=0



            pos_train_data = pos_train_data.sample(class_train_num,random_state = seeds[i])
            pos_test_data = pos_test_data.sample(class_train_num,random_state = seeds[i])

            neg_train_data = neg_train_data.sample(class_train_num,random_state = seeds[i])
            neg_test_data = neg_test_data.sample(class_train_num,random_state = seeds[i])



            data_train = pd.concat([pos_train_data,neg_train_data])
            data_test = pd.concat([pos_test_data,neg_test_data])

            #print('Preprocessing...')
            pdata_train = preprocess(data_train)
            pdata_test=preprocess(data_test)     

            for l in tqdm([0,1,2,3,4,5,6,7,8,9,10,11,12]):



                # get embeddings and prepare classification data

                #print('Inferring...')
                inf_data_train=infer(pdata_train,l)

                # train linear classifier (check repo)
                #cls = SGDClassifier(alpha=0.01,random_state=seeds[i])
                cls = SGDClassifier(alpha=0.01,random_state=seeds[1])

                cls.fit(list(inf_data_train.Xembed),inf_data_train.y)

                #test

                inf_data_test=infer(pdata_test,l)

                pred=cls.predict(list(inf_data_test.Xembed))

                #data_test['pred']=pred

                f1Score=f1_score(pred,inf_data_test.y)
                acc=accuracy_score(pred,inf_data_test.y)

                res[k][class_train_num][l]['f1'] = res[k][class_train_num][l].get('f1',[]) + [f1Score]
                res[k][class_train_num][l]['acc'] = res[k][class_train_num][l].get('acc',[]) + [acc]


                if f1Score>max_f1:
                  max_f1=f1Score
                  best_cls = cls 

        for l in tqdm([0,1,2,3,4,5,6,7,8,9,10,11,12]):
            res[k][class_train_num][l]['f1_mean'] = np.mean(res[k][class_train_num][l]['f1'])
            res[k][class_train_num][l]['f1_std'] = np.std(res[k][class_train_num][l]['f1'])

            res[k][class_train_num][l]['acc_mean'] = np.mean(res[k][class_train_num][l]['acc'])
            res[k][class_train_num][l]['acc_std'] = np.std(res[k][class_train_num][l]['acc'])
            print(f"{l}:{res[k][class_train_num][l]['f1_mean']}")

            res[k][class_train_num][l]['best_f1'] = max_f1
            res[k][class_train_num][l]['best_cls'] = best_cls


output_path = f"Layerwise/{type_}/{seed}/{max_k}/{train_sample_size}/"
Path(output_path).mkdir(parents=True, exist_ok=True)

for mult in res:
  for key in res[mult]:
      res_tup = sorted(res[mult][key].items(),key=lambda x: int(x[0]))
      layers = [x[0] for x in res_tup]
      vals = [x[1]['f1_mean'] for x in res_tup]
      plt.plot(layers,vals,label=f'k={mult}')
plt.xticks(layers,labels=layers)
plt.title(f'{key}')
plt.ylim([0,1])
plt.xlim([0,12])
#plt.savefig(f'{key}.png')   # <-- save first
plt.legend()
plt.savefig(f'{output_path}image.png', bbox_inches='tight')

#plt.show()


# with open(output_path+f"{datetime.now().strftime('%Y%m-%d%H-%M%S')}_res.json",'w') as f:
#     json.dump(res,f,indent=4)


with open(output_path+f"{datetime.now().strftime('%Y%m-%d%H-%M%S')}_res.pkl",'wb') as f:
    pickle.dump(res,f)

