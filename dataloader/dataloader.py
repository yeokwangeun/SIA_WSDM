import torch
import numpy as np
from torch.utils.data import DataLoader


class DataLoaderHandler:
    def __init__(self, mode, dataset, args, logger):
        self.mode = mode
        self.maxlen = args.maxlen
        self.batch_size = args.batch_size
        self.device = args.device
        self.logger = logger
        self.n_item_feats = len(dataset.item_feats)
        self.dataloader = self.load_dataloader(dataset)

    def load_dataloader(self, dataset):
        shuffle_mode = True if self.mode == "train" else False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_mode,
            collate_fn=self.collate_fn,
        )

    def get_dataloader(self):
        return self.dataloader

    def collate_fn(self, sample):
        seq_list, pos_list, next_item_list = [], [], []
        item_feat_lists = [[] for _ in range(self.n_item_feats)]
        for seq, next_item, i_feats in sample:
            seqlen = len(seq)
            pos = [i + 1 for i in range(seqlen)]
            if seqlen >= self.maxlen:
                seq = seq[:self.maxlen]
                pos = pos[:self.maxlen]
            else:
                padded = [0] * (self.maxlen - seqlen)
                seq = seq + padded
                pos = pos + padded
            seq_list.append(torch.tensor(seq))
            pos_list.append(torch.tensor(pos))
            next_item_list.append(torch.tensor(next_item))
            for i, i_feat in enumerate(i_feats):
                item_feat_lists[i].append(torch.tensor(i_feat, dtype=torch.float))
        seq_list = torch.stack(seq_list).to(self.device)
        pos_list = torch.stack(pos_list).to(self.device)
        next_item_list = torch.stack(next_item_list).to(self.device)
        return (seq_list, pos_list, next_item_list, *item_feat_lists)
