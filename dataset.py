import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
import torch
import os


class BubbleData(Dataset):
    def __init__(self, path_price_data, path_embed_data, load_embeds=True, len= None):
        with open(path_price_data, "rb") as f:
            self.data = pkl.load(f)
        self.len = len
        if load_embeds:
            with open(path_embed_data, "rb") as f:
                self.embed_data = pkl.load(f)

        self.load_embeds = load_embeds

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        if self.load_embeds:
            return (self.embed_data[idx]["embeddings"],
                    torch.tensor(data_dict["lookahead_starts"]),
                    torch.tensor(data_dict["lookahead_ends"]),
                    data_dict["n_bubbles"],
                    torch.tensor(data_dict["bubble"]),
                    self.embed_data[idx]["time_feats"],
                    torch.tensor(self.embed_data[idx]["len"]))

        else:
            return (torch.tensor(data_dict["lookback_price"]),
                    torch.tensor(data_dict["lookahead_starts"]),
                    torch.tensor(data_dict["lookahead_ends"]),
                    data_dict["n_bubbles"],
                    torch.tensor(data_dict["bubble"]))

    def __len__(self):
        if self.len != None:
            return self.len
        else: 
            return len(self.data)



class BubbleDatav2(Dataset):
    def __init__(self, price_data_path, embed_folder_path, load_embeds=True):

        self.folder_path = embed_folder_path
        self.load_embeds = load_embeds
        with open(price_data_path, "rb") as f:
            self.data = pkl.load(f)
        assert len(os.listdir(self.folder_path)) == len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if self.load_embeds:
            with open(f"{self.folder_path}/{idx}.pkl", "rb") as f:
                embed_data = pkl.load(f)
            return (embed_data["embeddings"],
                    torch.tensor(data_dict["lookahead_starts"]),
                    torch.tensor(data_dict["lookahead_ends"]),
                    data_dict["n_bubbles"],
                    torch.tensor(data_dict["bubble"]),
                    embed_data["time_feats"],
                    torch.tensor(embed_data["len"]))

        else:
            return (torch.tensor(data_dict["lookback_price"]),
                    torch.tensor(data_dict["lookahead_starts"]),
                    torch.tensor(data_dict["lookahead_ends"]),
                    data_dict["n_bubbles"],
                    torch.tensor(data_dict["bubble"]))

    def __len__(self):
        return len(self.data)
