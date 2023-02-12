import torch
import numpy as np
import random
from torch.utils.data import DataLoader


class DataLoaderHandler:
    def __init__(self, mode, dataset, args, logger):
        self.mode = mode
        self.maxlen = args.maxlen
        self.batch_size = args.batch_size
        self.device = args.device
        self.logger = logger
        self.dataset = dataset
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
        seq_list, pos_list, next_item_list, candidate_list = [], [], [], []
        item_feat_lists = [[] for _ in range(len(self.dataset.item_feats))]
        for seq, next_item, i_feats in sample:
            seqlen = len(seq)
            pos = [i + 1 for i in range(seqlen)]
            if seqlen >= self.maxlen:
                seq = seq[-self.maxlen :]
                pos = pos[: self.maxlen]
            else:
                padded = [0] * (self.maxlen - seqlen)
                seq = padded + seq
                pos = padded + pos
            seq_list.append(torch.tensor(seq))
            pos_list.append(torch.tensor(pos))
            next_item_list.append(torch.tensor(next_item))
            for i, i_feat in enumerate(i_feats):
                item_feat_lists[i].append(
                    torch.tensor(np.array(i_feat), dtype=torch.float).to(self.device)
                )
            if self.mode != "train" and self.dataset.eval_mode != "full":
                candidate = self.get_candidates(next_item, self.dataset.pop, self.dataset.eval_mode)
                candidate_list.append(candidate)
        seq_list = torch.stack(seq_list).to(self.device)
        pos_list = torch.stack(pos_list).to(self.device)
        next_item_list = torch.stack(next_item_list).to(self.device)
        candidate_list = torch.stack(candidate_list).to(self.device) if candidate_list else None
        if self.mode == "train":
            return (seq_list, pos_list, next_item_list, *item_feat_lists)
        else:
            return (seq_list, pos_list, next_item_list, candidate_list, *item_feat_lists)

    def get_candidates(self, next_item, pop, eval_mode):
        if eval_mode == "uni":
            candidates = tuple(random.sample(pop, 100))
        elif eval_mode == "pop":
            candidates = pop[:100]
        else:
            self.logger.error("Eval sampling mode should be either pop or uni.")
            raise Exception
        if next_item not in candidates:
            candidates = tuple([next_item]) + candidates[:-1]
        return torch.tensor(candidates)
