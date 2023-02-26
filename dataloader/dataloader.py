import torch
import numpy as np
import random
import math
from torch.utils.data import DataLoader


class DataLoaderHandler:
    def __init__(self, mode, dataset, args, logger):
        self.mode = mode
        self.maxlen = args.maxlen
        self.batch_size = args.batch_size
        self.device = args.device
        self.crop_random = args.crop_random
        self.crop_ratio = args.crop_ratio
        self.logger = logger
        self.dataset = dataset
        self.dataloader = self.load_dataloader(dataset)
        self.pad_item_feats = [np.zeros((self.dataset.n_neg, args.maxlen, dim)) for dim in args.dim_item_feats]

    def load_dataloader(self, dataset):
        if self.mode == "train":
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn_train,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=self.collate_fn_test,
            )

    def get_dataloader(self):
        return self.dataloader
    
    def collate_fn_train(self, sample):
        x_seq, x_p, y_pos_seq, y_pos_p, y_neg_seq, y_neg_p = [], [], [], [], [], []
        x_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_pos_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_neg_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        for x, y_pos, y_neg in sample:
            x_set = [x, x_seq, x_p, x_item_feats]
            y_pos_set = [y_pos, y_pos_seq, y_pos_p, y_pos_item_feats]
            y_neg_set = [y_neg, y_neg_seq, y_neg_p, y_neg_item_feats]
            for inputs, out_seq, out_p, out_item_feats in [x_set, y_pos_set, y_neg_set]:
                seq, pos, i_feats = self.make_fixed_length_seq(*inputs)
                out_seq.append(seq)
                out_p.append(pos)
                for i, i_feat in enumerate(i_feats):
                    out_item_feats[i].append(i_feat)
        x_out = [x_seq, x_p, x_item_feats]
        y_pos_out = [y_pos_seq, y_pos_p, y_pos_item_feats]
        y_neg_out = [y_neg_seq, y_neg_p, y_neg_item_feats]
        for out in [x_out, y_pos_out, y_neg_out]:
            out[0] = torch.stack(out[0]).to(self.device)
            out[1] = torch.stack(out[1]).to(self.device)
            out[2] = [torch.stack(i_feat).to(self.device) for i_feat in out[2]]
        return (x_out, y_pos_out, y_neg_out)

    def collate_fn_test(self, sample):
        x_seq, x_p, y_pos_seq, y_pos_p, y_neg_seq, y_neg_p = [], [], [], [], [], []
        x_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_pos_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_neg_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        for x, y_pos, y_neg in sample:
            x_set = [x, x_seq, x_p, x_item_feats, self.make_fixed_length_seq]
            y_pos_set = [y_pos, y_pos_seq, y_pos_p, y_pos_item_feats, self.make_fixed_length_seq]
            y_neg_set = [y_neg, y_neg_seq, y_neg_p, y_neg_item_feats, self.make_fixed_length_seq_batch]
            for inputs, out_seq, out_p, out_item_feats, make_fixed_fn in [x_set, y_pos_set, y_neg_set]:
                seq, pos, i_feats = make_fixed_fn(*inputs)
                out_seq.append(seq)
                out_p.append(pos)
                for i, i_feat in enumerate(i_feats):
                    out_item_feats[i].append(i_feat)
        x_out = [x_seq, x_p, x_item_feats]
        y_pos_out = [y_pos_seq, y_pos_p, y_pos_item_feats]
        y_neg_out = [y_neg_seq, y_neg_p, y_neg_item_feats]
        for out in [x_out, y_pos_out, y_neg_out]:
            out[0] = torch.stack(out[0]).to(self.device)
            out[1] = torch.stack(out[1]).to(self.device)
            out[2] = [torch.stack(i_feat).to(self.device) for i_feat in out[2]]
        return (x_out, y_pos_out, y_neg_out)        

    def make_fixed_length_seq(self, seq, i_feats):
        seqlen = len(seq)
        pos = np.array([i + 1 for i in range(seqlen)])
        if seqlen >= self.maxlen:
            seq = seq[-self.maxlen :]
            pos = pos[: self.maxlen]
            i_feats = [i_feat[-self.maxlen :,] for i_feat in i_feats]
        else:
            padded = np.zeros(self.maxlen - seqlen)
            padded2d = padded[:, np.newaxis]
            seq = np.append(padded, np.array(seq))
            pos = np.append(padded, np.array(pos))
            i_feats = [
                np.concatenate([np.repeat(padded2d, i_feat.shape[1], axis=1), i_feat], axis=0)
                for i_feat in i_feats
            ]
        seq = torch.tensor(seq, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.long)
        i_feats = [torch.tensor(i_feat, dtype=torch.float) for i_feat in i_feats]
        return (seq, pos, i_feats)
    
    def make_fixed_length_seq_batch(self, seq, i_feats):
        n_seq, seqlen = seq.shape
        pos = np.array([i + 1 for i in range(seqlen)])
        pos = np.repeat(pos.reshape(1, -1), n_seq, axis=0)
        if seqlen >= self.maxlen:
            seq = seq[:, -self.maxlen:]
            pos = pos[:, -self.maxlen:]
            i_feats = [i_feat[:, -self.maxlen:, :] for i_feat in i_feats]
        else:
            padded = np.zeros((n_seq, self.maxlen - seqlen))
            # padded3d = padded[:, :, np.newaxis]
            seq = np.concatenate([padded, seq], axis=1)
            pos = np.concatenate([padded, pos], axis=1)
            i_feats = [
                np.concatenate([pad[:, :(self.maxlen - seqlen), :], i_feat], axis=1)
                for pad, i_feat in zip(self.pad_item_feats, i_feats)
            ]
        seq = torch.tensor(seq, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.long)
        i_feats = [torch.tensor(i_feat, dtype=torch.float) for i_feat in i_feats]
        return (seq, pos, i_feats)