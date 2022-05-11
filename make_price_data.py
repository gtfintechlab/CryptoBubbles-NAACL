import os.path as osp
import os
import datetime as dt
import pickle as pkl
import pandas as pd
from tqdm import tqdm

stride = 3
lookahead = 10
lookback = 5

path = "price_data"
coins = os.listdir(path)


def get_start_end(ary):
    """
    ary: n_days
    
    sample = [0,1,1,1,0,1,1]
    start  = [0,1,0,0,0,1,0]
    end =    [0,0,0,1,0,0,1]
    """
    starts = [0]*lookahead
    ends = [0]*lookahead
    
    if ary[0] == 1:
        starts[0] = 1
    if ary[-1] == 1:
        ends[-1] = 1
        
    for i in range(lookahead-1):
        if ary[i+1]  - ary[i] == 1:
            starts[i+1] =1
        elif ary[i+1]  - ary[i] == -1:
            ends[i] = 1
            
    return starts, ends

train_samples_list = []
val_samples_list = []
test_samples_list = []

import datetime as dt
start = "2016-03-09"
end = "2021-04-07"
startt = dt.datetime.strptime(start, "%Y-%m-%d")
endt = dt.datetime.strptime(end, "%Y-%m-%d")


train = dt.timedelta(days= 1575)
val = dt.timedelta(days=(1575 + 160))
test = dt.timedelta(days=(1855))

train_start =  startt
train_end = startt + train
val_start = startt + train + dt.timedelta(days=1+lookahead)
val_end = startt + val
test_start = startt + val + dt.timedelta(days=1+lookahead)
test_end = startt + test


train_start= dt.datetime.strftime(train_start,  "%Y-%m-%d")
val_start= dt.datetime.strftime(val_start,  "%Y-%m-%d")
test_start= dt.datetime.strftime(test_start,  "%Y-%m-%d")
train_end = dt.datetime.strftime(train_end,  "%Y-%m-%d")
val_end = dt.datetime.strftime(val_end,  "%Y-%m-%d")
test_end = dt.datetime.strftime(test_end,  "%Y-%m-%d")


for coin in tqdm(coins):
    price_data = pd.read_csv(f"{path}/{coin}")
    price_data_dates = price_data["datetime"].tolist()
    price_data_dates.sort()
    if len(price_data_dates) == 0:
        continue
    ndate = dt.datetime.strptime(price_data_dates[0], "%Y-%m-%d")
    max_date= dt.datetime.strptime(price_data_dates[-1], "%Y-%m-%d") - dt.timedelta(days=(lookahead+lookback))
    while ndate < max_date:
        cdatestr = dt.datetime.strftime(ndate, "%Y-%m-%d")
        cdate = ndate
        lookbackdates = []
        lookback_price = []
        temp = cdate
        for _ in range(lookback):    
            temp = temp + dt.timedelta(days=1)
            lookbackdates.append(dt.datetime.strftime(temp, "%Y-%m-%d"))

        Flag = True
        for d in lookbackdates:
            if (d not in price_data_dates):
                ndate = cdate + dt.timedelta(days=1)
                Flag = False
                break
            else:
                lookback_price.append(price_data[price_data["datetime"] == d]["close_x"].values[0])
        if Flag:
            temp = cdate + dt.timedelta(days=lookback)
            lookaheaddates = []

            for _ in range(lookahead):
                temp= temp + dt.timedelta(days=1)
                lookaheaddates.append(dt.datetime.strftime(temp, "%Y-%m-%d"))
            Flag2 = True
            for d in lookaheaddates:
                if d not in price_data_dates:
                    ndate = cdate  + dt.timedelta(days=1)
                    Flag2=False
                    break
                    
            if Flag2:
                ndate = cdate + dt.timedelta(days=stride)
                lookahead_price = []
                lookahead_bubble= []
                
                for d in lookaheaddates:
                    lookahead_price.append(price_data[price_data["datetime"] == d]["close_x"].values[0])
                    lookahead_bubble.append(price_data[price_data["datetime"] == d]["label"].values[0])
                
                starts, ends = get_start_end(lookahead_bubble)
            
                if lookaheaddates[-1] <= val_start:
                    train_samples_list.append(
                        {"coin_name": coin.replace(".csv", ""),
                         "lookback_dates": lookbackdates,
                         "lookaheaddates": lookaheaddates,
                         "bubble": lookahead_bubble,
                         "lookback_price": lookback_price,
                         "lookahead_price": lookahead_price,
                         "lookahead_starts": starts, 
                         "lookahead_ends":ends,
                         "n_bubbles": sum(starts),
                        }
                    )
                elif lookaheaddates[-1] > val_start and lookaheaddates[-1] <= test_start:
                    val_samples_list.append(
                    {"coin_name": coin.replace(".csv", ""),
                         "lookback_dates": lookbackdates,
                         "lookaheaddates": lookaheaddates,
                         "bubble": lookahead_bubble,
                         "lookback_price": lookback_price,
                         "lookahead_price": lookahead_price,
                         "lookahead_starts": starts, 
                         "lookahead_ends":ends,
                         "n_bubbles": sum(starts),
                        })
                else:
                    test_samples_list.append(
                        {"coin_name": coin.replace(".csv", ""),
                         "lookback_dates": lookbackdates,
                         "lookaheaddates": lookaheaddates,
                         "bubble": lookahead_bubble,
                         "lookback_price": lookback_price,
                         "lookahead_price": lookahead_price,
                         "lookahead_starts": starts, 
                         "lookahead_ends":ends,
                         "n_bubbles": sum(starts),
                        }
                    )

with open(f"train_data_price_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl", "wb") as f:
    pkl.dump(train_samples_list, f)

with open( f"val_data_price_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl","wb") as f:
    pkl.dump(val_samples_list, f)

with open(f"test_data_price_only_lookback_{lookback}_lookahead_{lookahead}_stride_{stride}.pkl", "wb") as f:
    pkl.dump(test_samples_list, f)