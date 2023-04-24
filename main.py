import sys
import argparse
import logging
import os
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import nvidia_smi
from datetime import datetime

from dataset import load_data, TrainDataset, EvalDataset
from dataloader import DataLoaderHandler
from model import SIA
from trainer import train, evaluate, WarmupBeforeMultiStepLR

BASEDIR = os.path.dirname(os.path.realpath(__file__))


def main():
    args = parse_arguments()
    log_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    log_dir = args.log_dir if args.log_dir else log_time
    log_dir = os.path.join(BASEDIR, os.path.join(f"log/{args.dataset}", log_dir))
    logger = get_logger(args.dataset, log_dir)
    logger.info(args)

    logger.info("Set Seed")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("Set GPU")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_count = torch.cuda.device_count()
    logger.info(f"{gpu_count} gpus found")
    if gpu_count > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(i) for i in range(3)])

    logger.info("Load Data")
    raw_dir = os.path.join(BASEDIR, "dataset/raw")
    processed_dir = os.path.join(BASEDIR, "dataset/processed")
    inter, item_feats, pop, cold_inter = load_data(args, raw_dir, processed_dir, logger)
    dim_item_feats = [tuple(feat.values())[0].shape[0] for feat in item_feats]
    args.dim_item_feats = dim_item_feats
    args.num_items = len(pop)

    logger.info("Loading Dataloaders")
    train_dataset = TrainDataset(inter, item_feats, pop, args, logger)
    val_dataset = EvalDataset(
        inter, item_feats, pop, args, logger, mode="val", eval_mode=args.eval_sample_mode,
    )
    test_dataset = EvalDataset(
        inter, item_feats, pop, args, logger, mode="test", eval_mode=args.eval_sample_mode,
    )
    cold_dataset = EvalDataset(
        cold_inter, item_feats, pop, args, logger, mode="test", eval_mode=args.eval_sample_mode,
    )
    train_loader = DataLoaderHandler("train", train_dataset, args, logger).get_dataloader()
    val_loader = DataLoaderHandler("val", val_dataset, args, logger).get_dataloader()
    test_loader = DataLoaderHandler("test", test_dataset, args, logger).get_dataloader()
    cold_loader = DataLoaderHandler("test", cold_dataset, args, logger).get_dataloader()

    logger.info("Loading Model")
    model = SIA(
        fusion_mode=args.seq_fusion_mode,
        out_token=args.out_token,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        attn_num_heads=args.attn_num_heads,
        attn_dim_head=args.attn_dim_head,
        attn_depth=args.attn_depth,
        attn_self_per_cross=args.attn_self_per_cross,
        attn_dropout=args.attn_dropout,
        attn_ff_dropout=args.attn_ff_dropout,
        feat_mask_ratio=args.feat_mask_ratio,
        dim_item_feats=dim_item_feats,
        num_items=args.num_items,
        maxlen=args.maxlen,
        device=args.device,
    )
    if gpu_count > 1:
        model = nn.DataParallel(model)

    nvidia_smi.nvmlInit()
    if args.mode == "train":
        logger.info("Train the model")
        start_epoch = 0
        model = model.to(args.device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        writer = get_writer(args, log_dir)
        model = train(
            start_epoch,
            args.num_epochs,
            args.early_stop,
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            args.criterion,
            writer,
            logger,
            log_dir,
            args.device,
            args.item_fusion_mode,
            args.one_to_one_loss,
            args.eval_step,
        )
    else:
        if not args.saved_model_path:
            logger.error("For evaluation mode, saved model path must be given.")
            return 1
        model.load_state_dict(torch.load(args.saved_model_path))
        model = model.to(args.device)

    logger.info("Evaluation starts")
    test_metrics = evaluate(model, test_loader, args.item_fusion_mode)
    test_log = ""
    for k, v in test_metrics.items():
        test_log += f"{k}: {v:.5f} "
    logger.info(f"Test - {test_log}")

    logger.info("Evaluation on Cold Items starts")
    test_metrics = evaluate(model, cold_loader, args.item_fusion_mode)
    test_log = ""
    for k, v in test_metrics.items():
        test_log += f"{k}: {v:.5f} "
    logger.info(f"Cold Test - {test_log}")

    if args.mode == "train":
        writer.add_hparams(hparam_dict={"log_dir": args.log_dir}, metric_dict={k: v for k, v in test_metrics.items() if k != "NDCG@1"})        
        writer.flush()
        writer.close()
    nvidia_smi.nvmlShutdown()


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
    parser.add_argument("--cold_item_threshold", type=int, default=1)
    
    #################### MODEL ####################
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--feat_mask_ratio", type=float, default=0.)    
    parser.add_argument("--attn_dim_head", type=int, default=64)
    parser.add_argument("--attn_num_heads", type=int, default=1)
    parser.add_argument("--attn_depth", type=int, default=1)
    parser.add_argument("--attn_self_per_cross", type=int, default=6)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--attn_ff_dropout", type=float, default=0.2)
    parser.add_argument("--seq_fusion_mode", type=str, default="not")
    parser.add_argument("--item_fusion_mode", type=str, default="mean")
    parser.add_argument("--out_token", type=str, default="cls")

    #################### TRAIN ####################
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_step", type=int, default=None)
    parser.add_argument("--lr_milestones", type=list, default=None)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--sequence_split", type=int, default=1)
    parser.add_argument("--one_to_one_loss", type=int, default=0)
    parser.add_argument("--criterion", type=str, default="BCE")
    parser.add_argument("--n_negs_train", type=int, default=1)

    #################### EVALUATION ####################
    parser.add_argument("--eval_sample_mode", type=str, default="uni")
    parser.add_argument("--eval_step", type=int, default=1)

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


if __name__ == "__main__":
    sys.exit(main())
