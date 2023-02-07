import sys
import argparse
import logging
import os
import torch
import numpy as np
import random
from datetime import datetime

from dataset import load_data, TrainDataset, EvalDataset
from dataloader import DataLoaderHandler

BASEDIR = os.path.dirname(os.path.realpath(__file__))


def main():
    args = parse_arguments()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(args.dataset, os.path.join(BASEDIR, args.log_dir))
    logger.info(args)

    logger.info("Set Seed")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("Load Data")
    raw_dir = os.path.join(BASEDIR, "dataset/raw")
    processed_dir = os.path.join(BASEDIR, "dataset/processed")
    inter, item_feat, pop = load_data(args, raw_dir, processed_dir, logger)

    logger.info("Loading Dataloaders")
    train_dataset = TrainDataset(inter, args, logger)
    val_dataset = EvalDataset(inter, args, logger, mode="val")
    test_dataset = EvalDataset(inter, args, logger, mode="test")
    train_loader = DataLoaderHandler("train", train_dataset, args, logger).get_dataloader()
    val_loader = DataLoaderHandler("val", val_dataset, args, logger).get_dataloader()
    test_loader = DataLoaderHandler("test", test_dataset, args, logger).get_dataloader()

    logger.info("Loading Model")
    # model = SIA(item_feat)


def parse_arguments():
    parser = argparse.ArgumentParser()
    #################### GLOBAL ####################
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_dir", type=str, default="log")

    #################### DATA ####################
    parser.add_argument("--dataset", type=str, default="amazon_beauty")
    parser.add_argument("--content", type=list, default=["image", "desc"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=50)

    #################### MODEL ####################

    #################### TRAIN ####################
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    #################### EVALUATION ####################
    parser.add_argument("--eval_sample_mode", type=str, default="uni")

    return parser.parse_args()


def get_logger(dataset_name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = f"{dataset_name}_{datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    sys.exit(main())
