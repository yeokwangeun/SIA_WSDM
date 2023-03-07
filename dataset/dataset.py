import os
import pickle
import pandas as pd
import numpy as np
import copy
import random
from tqdm import tqdm


def load_data(args, raw_dir, processed_dir, logger):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    inter_file = os.path.join(processed_dir, f"{args.dataset}_inter.pkl")
    item_file = os.path.join(processed_dir, f"{args.dataset}_item.pkl")
    pop_file = os.path.join(processed_dir, f"{args.dataset}_pop.pkl")
    if os.path.exists(inter_file) and os.path.exists(item_file) and os.path.exists(pop_file):
        logger.info("Preprocessing already done")
        with open(inter_file, "rb") as inter_pf, open(item_file, "rb") as item_pf, open(
            pop_file, "rb"
        ) as pop_pf:
            inter = pickle.load(inter_pf)
            item = pickle.load(item_pf)
            pop = pickle.load(pop_pf)
        return (inter, item, pop)

    logger.info("Preprocessing starts")
    raw_dir = os.path.join(raw_dir, args.dataset)
    ratings = pd.read_csv(os.path.join(raw_dir, f"ratings_{args.dataset}.csv"))
    ratings.columns = ["user_id", "item_id", "timestamp"]
    item = []
    for content in args.content:
        with open(os.path.join(raw_dir, f"{args.dataset}_{content}_features.pkl"), "rb") as pf:
            feat = pickle.load(pf)
            if content == "image":
                feat = {k: np.mean(v, axis=0) for k, v in feat.items()}
            item.append(feat)

    logger.info("Mapping user_id and item_id to index")
    ratings, *item = map_to_index(ratings, *item)
    logger.info("Make sequence shaped data")
    inter = convert_to_seq(ratings)
    logger.info("Get popularity information")
    pop = get_popularity(ratings)

    logger.info("Preprocessing done")
    with open(inter_file, "wb") as inter_pf, open(item_file, "wb") as item_pf, open(
        pop_file, "wb"
    ) as pop_pf:
        pickle.dump(inter, inter_pf)
        pickle.dump(item, item_pf)
        pickle.dump(pop, pop_pf)

    return (inter, item, pop)


def map_to_index(ratings, *item_feat):
    user_mapper = {uid: (i + 1) for i, uid in enumerate(ratings["user_id"].unique())}
    item_mapper = {iid: (i + 1) for i, iid in enumerate(ratings["item_id"].unique())}

    ratings["user_id"] = [user_mapper[uid] for uid in ratings["user_id"]]
    ratings["item_id"] = [item_mapper[iid] for iid in ratings["item_id"]]
    item_feat = [{item_mapper[k]: v for k, v in feat.items()} for feat in item_feat]
    return (ratings, *item_feat)


def convert_to_seq(ratings):
    ratings.sort_values(by="timestamp", inplace=True)
    inter = ratings.groupby("user_id")["item_id"].apply(list).reset_index(name="items")
    return inter


def get_popularity(ratings):
    pop = ratings.groupby("item_id").count()["user_id"].reset_index(name="count")
    pop.sort_values(by="count", ascending=False, inplace=True)
    return tuple(pop["item_id"].to_list())


class TrainDataset:
    def __init__(self, inter, item_feats, pop, args, logger):
        self.df = pd.DataFrame({"items": [items[:-2] for items in inter["items"]]})
        split_point = 2 if args.sequence_split else args.maxlen
        self.df = pd.DataFrame(
            {
                "items": [
                    seq[: i + 1]
                    for seq in self.df["items"]
                    for i in range(min(len(seq), split_point) - 1, len(seq))
                ]
            }
        )
        self.df["tmp"] = [" ".join([str(item) for item in items]) for items in self.df["items"]]
        self.df.drop_duplicates(subset=["tmp"], inplace=True)
        self.df["seq"] = [items[:-1] for items in self.df["items"]]
        self.df["next_item"] = [items[-1] for items in self.df["items"]]
        self.df["next_neg"] = [[i for i in random.choices(pop, k=2) if i != item][0] for item in self.df["next_item"]]
        self.df = self.df[["seq", "next_item", "next_neg"]]
        self.item_feats = item_feats
        self.args = args
        self.logger = logger
        self.n_neg = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = np.array(self.df.iloc[idx, 0])
        next_item = self.df.iloc[idx, 1]
        if random.random() < self.args.hard_negative_ratio:
            next_neg = random.choice(seq)
            if next_neg == next_item:
                next_neg = self.df.iloc[idx, 2]
        else:
            next_neg = self.df.iloc[idx, 2]
        
        item_feats = []
        item_feats_pos = []
        item_feats_neg = []
        for mapper in self.item_feats:
            seq_feat = np.array([mapper[item_id] for item_id in seq])
            pos_feat = np.array(mapper[next_item])
            if random.random() < self.args.hard_negative_feature_ratio:
                neg_feat = pos_feat
            else:
                neg_feat = np.array(mapper[next_neg])
            item_feats.append(seq_feat)
            item_feats_pos.append(pos_feat)
            item_feats_neg.append(neg_feat)

        return ((seq, item_feats), (next_item, item_feats_pos), (next_neg, item_feats_neg))


class EvalDataset:
    def __init__(self, inter, item_feats, pop, args, logger, mode, eval_mode, n_neg=99):
        self.item_feats = item_feats
        self.pop = pop
        self.args = args
        self.logger = logger
        self.eval_mode = eval_mode
        self.n_neg = n_neg
        self.df = copy.deepcopy(inter)
        last_idx = -2 if mode == "val" else -1
        self.df["seq"] = [items[:last_idx] for items in self.df["items"]]
        self.df["next_item"] = [items[last_idx] for items in self.df["items"]]
        self.logger.info("Get candidates for evaluation")
        self.df["next_negs"] = [self.get_candidates(next_item) for next_item in tqdm(self.df["next_item"])]
        self.df = self.df[["seq", "next_item", "next_negs"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = np.array(self.df.iloc[idx, 0])
        next_item = self.df.iloc[idx, 1]
        next_negs = np.array(self.df.iloc[idx, 2])

        item_feats = []
        item_feats_pos = []
        item_feats_neg = []
        for mapper in self.item_feats:
            seq_feat = np.array([mapper[item_id] for item_id in seq])
            pos_feat = np.array(mapper[next_item])
            neg_feats = np.array([mapper[next_neg] for next_neg in next_negs])
            item_feats.append(seq_feat)
            item_feats_pos.append(pos_feat)
            item_feats_neg.append(neg_feats)

        return ((seq, item_feats), (next_item, item_feats_pos), (next_negs, item_feats_neg))

    def get_candidates(self, next_item):
        pool = list(self.pop)
        pool.remove(next_item)
        if self.eval_mode == "full":
            candidates = pool
        elif self.eval_mode == "uni":
            candidates = random.sample(pool, self.n_neg)
        elif self.eval_mode == "pop":
            candidates = pool[:self.n_neg]
        else:
            self.logger.error(f"Eval sampling mode in wrong form: {self.eval_mode}")
            raise Exception
        return candidates