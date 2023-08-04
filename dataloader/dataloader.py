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
        self.dataset = dataset
        self.dataloader = self.load_dataloader(dataset)
        self.pad_item_feats = [np.zeros((self.dataset.n_neg, args.maxlen, dim)) for dim in args.dim_item_feats]
        self.num_items = args.num_items

    def load_dataloader(self, dataset):
        if self.mode == "train":
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

    def get_dataloader(self):
        return self.dataloader

    def collate_fn(self, sample):
        x_seq, x_p, y_pos, y_neg = [], [], [], []
        x_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_pos_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        y_neg_item_feats = [[] for _ in range(len(self.dataset.item_feats))]
        for x, next_item, next_neg in sample:
            seq, pos, i_feats = self.make_fixed_length_seq(*x)
            x_seq.append(seq)
            x_p.append(pos)
            for i, i_feat in enumerate(i_feats):
                x_item_feats[i].append(i_feat)
            y_pos.append(torch.tensor(next_item[0], dtype=torch.long))
            y_neg.append(torch.tensor(next_neg[0], dtype=torch.long))
            for i in range(len(y_pos_item_feats)):
                y_pos_item_feats[i].append(torch.tensor(next_item[1][i], dtype=torch.float))
                y_neg_item_feats[i].append(torch.tensor(next_neg[1][i], dtype=torch.float))

        x_out = [x_seq, x_p, x_item_feats]
        y_pos_out = [y_pos, y_pos_item_feats]
        y_neg_out = [y_neg, y_neg_item_feats]
        x_out[0] = torch.stack(x_out[0]).to(self.device)
        x_out[1] = torch.stack(x_out[1]).to(self.device)
        x_out[2] = [torch.stack(i_feat).to(self.device) for i_feat in x_out[2]]
        for y_out in [y_pos_out, y_neg_out]:
            y_out[0] = torch.stack(y_out[0]).to(self.device)
            y_out[1] = [torch.stack(i_feat).to(self.device) for i_feat in y_out[1]]
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
            i_feats = [np.concatenate([np.repeat(padded2d, i_feat.shape[1], axis=1), i_feat], axis=0) for i_feat in i_feats]
        seq = torch.tensor(seq, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.long)
        i_feats = [torch.tensor(i_feat, dtype=torch.float) for i_feat in i_feats]
        return (seq, pos, i_feats)
