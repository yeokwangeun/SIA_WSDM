import sys
import argparse
import logging
import os
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from datetime import datetime

from dataset import load_data, TrainDataset, EvalDataset
from dataloader import DataLoaderHandler
from model import SIA
from trainer import train, evaluate, WarmupBeforeMultiStepLR

BASEDIR = os.path.dirname(os.path.realpath(__file__))


def main():
    args = parse_arguments()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    log_dir = args.log_dir if args.log_dir else log_time
    log_dir = os.path.join(BASEDIR, os.path.join(f"log/{args.dataset}", log_dir))
    logger = get_logger(args.dataset, log_dir)
    logger.info(args)

    logger.info("Set Seed")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("Load Data")
    raw_dir = os.path.join(BASEDIR, "dataset/raw")
    processed_dir = os.path.join(BASEDIR, "dataset/processed")
    inter, item_feats, pop = load_data(args, raw_dir, processed_dir, logger)
    dim_item_feats = [tuple(feat.values())[0].shape[0] for feat in item_feats]
    num_items = len(pop)

    logger.info("Loading Dataloaders")
    train_dataset = TrainDataset(inter, item_feats, args, logger)
    val_dataset = EvalDataset(
        inter, item_feats, pop, args, logger, mode="val", eval_mode=args.eval_sample_mode
    )
    test_dataset = EvalDataset(
        inter, item_feats, pop, args, logger, mode="test", eval_mode=args.eval_sample_mode
    )
    train_loader = DataLoaderHandler("train", train_dataset, args, logger).get_dataloader()
    val_loader = DataLoaderHandler("val", val_dataset, args, logger).get_dataloader()
    test_loader = DataLoaderHandler("test", test_dataset, args, logger).get_dataloader()

    logger.info("Loading Model")
    model = SIA(
        latent_dim=args.latent_dim,
        item_num_outputs=args.item_num_outputs,
        item_num_heads=args.item_num_heads,
        item_num_latents=args.item_num_latents,
        item_dim_hidden=args.item_dim_hidden * args.item_num_heads,
        attn_depth=args.attn_depth,
        attn_self_per_cross=args.attn_self_per_cross,
        attn_dropout=args.attn_dropout,
        attn_ff_dropout=args.attn_ff_dropout,
        attn_num_heads=args.attn_num_heads,
        attn_dim_head=args.attn_dim_head,
        dim_item_feats=dim_item_feats,
        num_items=num_items,
        maxlen=args.maxlen,
        device=args.device,
    )

    if args.mode == "train":
        logger.info("Train the model")
        start_epoch = 0
        model = model.to(args.device)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )
        scheduler = WarmupBeforeMultiStepLR(
            optimizer,
            warmup_step=args.lr_warmup_step,
            milestones=args.lr_milestones,
            gamma=args.lr_gamma,
        )
        chkpoint_path = f"{log_dir}/checkpoint.pt"
        if os.path.exists(chkpoint_path):
            chkpoint = torch.load(chkpoint_path)
            start_epoch = chkpoint["last_epoch"] + 1
            model.load_state_dict(chkpoint["model_state_dict"])
            optimizer.load_state_dict(chkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(chkpoint["scheduler_state_dict"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        writer = get_writer(args, log_dir)
        model, last_epoch = train(
            start_epoch,
            args.num_epochs,
            args.early_stop,
            train_loader,
            val_loader,
            args.eval_sample_mode,
            num_items,
            model,
            optimizer,
            scheduler,
            loss_fn,
            writer,
            logger,
        )
        save_dict = {
            'last_epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        save_model(log_dir, save_dict)
    else:
        if not args.saved_model_path:
            logger.error("For evaluation mode, saved model path must be given.")
            return 1
        model.load_state_dict(torch.load(args.saved_model_path))

    logger.info("Evaluation starts")
    test_metrics = evaluate(model, test_loader, args.eval_sample_mode, num_items)
    test_log = ""
    for k, v in test_metrics.items():
        test_log += f"{k}: {v:.5f} "
    logger.info(f"Test - {test_log}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    #################### GLOBAL ####################
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_dir", type=str, default=None)

    #################### DATA ####################
    parser.add_argument("--dataset", type=str, default="amazon_beauty")
    parser.add_argument("--content", type=list, default=["image", "desc"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=50)

    #################### MODEL ####################
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--item_dim_hidden", type=int, default=64)
    parser.add_argument("--item_num_heads", type=int, default=1)
    parser.add_argument("--attn_dim_head", type=int, default=64)
    parser.add_argument("--attn_num_heads", type=int, default=1)

    parser.add_argument("--item_num_outputs", type=int, default=4)
    parser.add_argument("--item_num_latents", type=int, default=4)
    parser.add_argument("--attn_depth", type=int, default=1)
    parser.add_argument("--attn_self_per_cross", type=int, default=6)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--attn_ff_dropout", type=float, default=0.2)

    #################### TRAIN ####################
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_step", type=int, default=None)
    parser.add_argument("--lr_milestones", type=list, default=None)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--early_stop", type=int, default=50)

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
    filename = f"{dataset_name}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


def get_writer(args, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        f.writelines([f"{k}: {v}\n" for k, v in vars(args).items()])
    writer = SummaryWriter(log_dir)
    return writer


def save_model(save_dir, save_dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, "model.pt")
    chkpoint_save_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(save_dict["model_state_dict"], model_save_path)
    torch.save(save_dict, chkpoint_save_path)


if __name__ == "__main__":
    sys.exit(main())
