import os.path as osp
import os
import datetime as dt
import pickle as pkl
import torch
import pandas as pd
import datetime
import random
import numpy as np
from tqdm import tqdm_notebook
import os
import shutil
import pandas as pd
from tqdm import notebook
import torch
from transformers import AutoModel, AutoTokenizer 
from collections import defaultdict
import pickle as pkl
import random
import argparse


parser = argparse.ArgumentParser(
    description="Text Data Creator")

parser.add_argument(
    "--split",
    default="train",
    type=str,
    help="Data split to be used (default: train)",
)

parser.add_argument(
    "--num_lookback",
    default=5,
    type=int,
    help="Number of Lookback days (default: 5)",
)

parser.add_argument(
    "--num_lookahead",
    default=10,
    type=int,
    help="Number of Lookahead days (default: 10)",
)

parser.add_argument(
    "--stride",
    default=3,
    type=int,
    help="Stride to use while creating the data (default: 3)",
)


args = parser.parse_args()


split = args.split
stride = args.stride
lookback = args.num_lookback
lookahead = args.num_lookahead


if os.path.exists("tweet_data_embed") == False:
    os.mkdir("tweet_data_embed")

with open(f"{split}_data_price_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl", "rb") as f:
    data = pkl.load(f)

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
device = torch.device("cuda:0") 
bertweet = bertweet.to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

def get_embeddings(tweets):
    encoding = tokenizer(tweets, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    norm_weights =  torch.nn.functional.normalize(attention_mask.float(), p=1, dim=1)
    outputs = bertweet(input_ids, attention_mask=attention_mask)
    last_hidden_states = outputs[0].detach()
    weighted_hidden_states = norm_weights.unsqueeze(2)*last_hidden_states
    avg_last = torch.sum(weighted_hidden_states, dim=1)
    avg_last = avg_last.detach().cpu()
    
    return avg_last



new_data = []
new_data_embedding_len_time = []
not_done_idx = []
embedding_dict= {}
time_feature_dict = {}
black_list= []
c1= []
c2= []
c3 = []
sss= 0


for idx in tqdm_notebook(len(data)):
    sample = data[idx]
    coin = sample["coin_name"]
    date_list = sample["lookback_dates"]
    date_list.sort()
    embeddings = torch.zeros(15*lookback, 768)
    time_diff = torch.ones(15*lookback, 1)

    tweets = []
    time = []
    embeds = []

    for date in date_list:
        if f"{coin}_{date}" not in black_list:
            date_path = f"tweet_data/{coin}/{date}.csv"
            if os.path.exists(date_path):
                tweet_list= []
                created_at = []
                try:
                    date_data = pd.read_csv(date_path)
                except:
                    c1.append(date_path)
                    continue

                if "tweet" in date_data:
                    try:
                        tweet_list = date_data["tweet"].tolist()
                        created_at = date_data["created_at"].tolist()

                        if(len(tweet_list) > 15):
                            indices = sorted(list(np.random.choice(list(range(0,len(created_at))), replace=False, size=15)))
                            tweet_list= [tweet_list[x] for x in indices]
                            created_at = [created_at[x] for x in indices]
                        created_at_new=[]
                        for dt in created_at:
                            created_at_new.append(datetime.datetime.strptime(dt[:-4], "%Y-%m-%d %H:%M:%S"))
                    except:
                        continue

                elif "Text" in date_data:
                    try: 
                        tweet_list = date_data["Text"].tolist()
                        created_at = date_data["Timestamp"].tolist()

                        if(len(tweet_list) > 15):
                            indices = sorted(list(np.random.choice(list(range(0,len(created_at))), replace=False, size=15)))
                            tweet_list= [tweet_list[x] for x in indices]
                            created_at = [created_at[x] for x in indices]

                        created_at_new=[]
                        for dt in created_at:
                            dt = dt[:-5]
                            dt = dt.replace("T", " ")
                            created_at_new.append(datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S"))
                    except:
                        continue

                if len(tweet_list) > 0:
                    sample_embed = get_embeddings(tweet_list).detach().cpu()
                    with open(f"tweet_data_embed/{split}/{coin}_{date}.pkl", "wb") as f:
                        pkl.dump({
                            "embedding": sample_embed,
                            "created_at": created_at_new
                        },f)

                    embeds.append(sample_embed)
                    time.extend(created_at_new)
                else:
                    black_list.append(f"{coin}_{date}")

    if len(embeds) > 0:
        created_at = np.array(time)
        indices = np.argsort(created_at)
        created_at = created_at[indices]

        batched_embeddings = torch.cat(embeds, dim=0)
        batched_embeddings = batched_embeddings[indices]

        j = 15*lookback-len(created_at)+1
        for i in range(1, len(created_at)):
            # calc time difference in minutes
            delta_time = created_at[i] - created_at[i - 1]
            delta_min = int(delta_time.total_seconds() / 60)
            # if not the same min
            if delta_min != 0:
                time_diff[j] = 1 / delta_min
            j+=1

        embeddings[15*lookback-len(created_at):] = batched_embeddings
        new_data_embedding_len_time.append(
        {
            "embeddings": embeddings, 
            "time_feats": time_diff,
            "len": len(created_at)
        })
        new_data.append(sample)

    else:
        not_done_idx.append(idx)
        
with open(f"{split}_data_price_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl", "wb") as f:
    pkl.dump(new_data, f)

with open(f"{split}_data_text_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl", "wb") as f:
    pkl.dump(new_data_embedding_len_time, f)