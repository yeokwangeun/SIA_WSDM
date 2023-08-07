import sys
import argparse
import logging
import os
import yaml
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
    log_dir = os.path.join(BASEDIR, os.path.join(f"log/{args.dataset}", args.exp_name))
    logger = get_logger(args.dataset, log_dir)
    logger.info(args)

    logger.info("Set Seed")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info(f"Set GPU: {args.gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    logger.info("Load Data")
    raw_dir = os.path.join(BASEDIR, "dataset/raw")
    processed_dir = os.path.join(BASEDIR, "dataset/processed")
    inter, item_feats, pop = load_data(args, raw_dir, processed_dir, logger)
    args.dim_item_feats = [tuple(feat.values())[0].shape[0] for feat in item_feats]
    args.num_items = len(pop)

    logger.info("Loading Dataloaders")
    train_dataset = TrainDataset(inter, item_feats, pop, args, logger)
    val_dataset = EvalDataset(inter, item_feats, pop, args, logger, mode="val", eval_mode=args.eval_sample_mode)
    test_dataset = EvalDataset(inter, item_feats, pop, args, logger, mode="test", eval_mode=args.eval_sample_mode)
    train_loader = DataLoaderHandler("train", train_dataset, args, logger).get_dataloader()
    val_loader = DataLoaderHandler("val", val_dataset, args, logger).get_dataloader()
    test_loader = DataLoaderHandler("test", test_dataset, args, logger).get_dataloader()

    logger.info("Loading Model")
    model = SIA(
        attn_mode=args.attn_mode,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        attn_num_heads=args.attn_num_heads,
        attn_dim_head=args.attn_dim_head,
        attn_depth=args.attn_depth,
        attn_self_per_cross=args.attn_self_per_cross,
        attn_dropout=args.attn_dropout,
        attn_ff_dropout=args.attn_ff_dropout,
        dim_item_feats=args.dim_item_feats,
        num_items=args.num_items,
        maxlen=args.maxlen,
        device=args.device,
        latent_random=args.latent_random,
        latent_with_pos=args.latent_with_pos,
        item_with_pos=args.item_with_pos,
    )
    if len(args.gpus.split(",")) > 1:
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
            args.eval_step,
        )
    else:
        if not args.saved_model_path:
            logger.error("For evaluation mode, saved model path must be given.")
            return 1
        model.load_state_dict(torch.load(args.saved_model_path))
        model = model.to(args.device)

    logger.info("Evaluation starts")
    test_metrics = evaluate(model, test_loader, args.item_fusion_mode, analysis_path=args.analysis_path)
    test_log = ""
    for k, v in test_metrics.items():
        test_log += f"{k}: {v:.5f} "
    logger.info(f"Test - {test_log}")
    if args.mode == "train":
        writer.add_hparams(hparam_dict={"log_dir": log_dir}, metric_dict={k: v for k, v in test_metrics.items() if k != "NDCG@1"})
        writer.flush()
        writer.close()
    nvidia_smi.nvmlShutdown()


def parse_arguments():
    str2list = lambda input_str: input_str.replace(" ", "").split(",")

    def str2bool(input_str):
        input_str = input_str.strip().lower()
        if input_str in ("true", "y", "yes", "1"):
            return True
        elif input_str in ("false", "n", "no", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    #################### GLOBAL ####################
    parser.add_argument("--config_file", type=str, default=None, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Determines the mode of operation, either 'train' for training or 'test' for testing the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed number for random number generation, useful for reproducibility.")
    parser.add_argument(
        "--exp_name", type=str, default=datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"), help="Name of the experiment, used to identify and save results with a timestamp by default."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device to run the code on, either 'cuda' for GPU or 'cpu' for CPU.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=", ".join([str(i) for i in range(torch.cuda.device_count())]),
        help="Comma-separated list of GPU device indices to be used for distributed training or multiple GPUs. By default, it uses all available GPUs.",
    )
    parser.add_argument("--saved_model_path", type=str, default=None, help="Path to a pre-trained model for testing.")
    parser.add_argument("--analysis_path", type=str, default=None, help="Path to evaluation results for analysis.")

    #################### DATA ####################
    parser.add_argument("--dataset", type=str, default="amazon_beauty", choices=["amazon_beauty", "amazon_sports", "amazon_toys", "ml-1m"], help="Specifies the dataset to use.")
    parser.add_argument(
        "--content",
        type=str2list,
        default="image,desc",
        help="Specifies the content types to use, as a comma-separated list. For example, 'image,desc' means using both image and text description feature.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing data.")
    parser.add_argument("--maxlen", type=int, default=50, help="Maximum sequence length.")
    parser.add_argument("--sequence_split", type=str2bool, default="Yes", help="Split training sequences into subsequences if set to True, otherwise use the complete sequences as is.")

    #################### MODEL ####################
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimensionality of the latent vector.")
    parser.add_argument("--feature_dim", type=int, default=64, help="Dimensionality of the content features.")
    parser.add_argument("--attn_dim_head", type=int, default=64, help="Dimensionality of each attention head in the cross-attention and self-attention module.")
    parser.add_argument("--attn_num_heads", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--attn_depth", type=int, default=1, help="Specifies the number of iterations for the iterative attention mechanism.")
    parser.add_argument("--attn_self_per_cross", type=int, default=1, help="Determines the number of self-attention layers applied after cross-attention layer.")
    parser.add_argument("--attn_dropout", type=float, default=0.2, help="Dropout rate for attention layers.")
    parser.add_argument("--attn_ff_dropout", type=float, default=0.2, help="Dropout rate for feed-forward layers in the attention mechanism.")
    parser.add_argument("--attn_mode", type=str, default="masked", choices=["full", "masked"], help="Type of attention used, either 'full' for full attention or 'masked' for masked attention.")
    parser.add_argument("--latent_random", type=str2bool, default="No", help="Whether to initialize the latent space randomly. 'No' means using id embedding for latent vector.")
    parser.add_argument("--latent_with_pos", type=str2bool, default="Yes", help="Whether to include positional encoding in the latent space.")
    parser.add_argument("--item_with_pos", type=str2bool, default="Yes", help="Whether to include positional encoding in the item features.")
    parser.add_argument("--item_fusion_mode", type=str, default="mean", choices=["pure", "mean", "attn"], help="Specifies how item features are fused to make item representation.")

    #################### TRAIN ####################
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training.")
    parser.add_argument("--lr_warmup_step", type=int, default=None, help="Number of warm-up steps for the learning rate scheduler.")
    parser.add_argument("--lr_milestones", type=list, default=None, help="List of epochs at which to decrease the learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="Factor by which to decrease the learning rate at each milestone.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay used in the optimizer.")
    parser.add_argument("--early_stop", type=int, default=0, help="Number of epochs to wait before early stopping if no improvement in validation performance (NDCG@10).")
    parser.add_argument("--criterion", type=str, default="BCE", choices=["BCE", "CL"], help="Loss function used for training, either Binary Cross-Entropy or Contrastive-Learning.")
    parser.add_argument("--n_negs_train", type=int, default=1, help="Number of negative samples per positive sample during training.")

    #################### EVALUATION ####################
    parser.add_argument(
        "--eval_sample_mode",
        type=str,
        default="uni",
        choices=["uni", "pop", "full"],
        help="Sample mode for evaluation, either 'uni' (uniform random), 'pop' (popularity-based), or 'full' (full evaluation).",
    )
    parser.add_argument("--eval_step", type=int, default=1, help="Frequency of evaluation during training (e.g., evaluate every 'eval_step' epochs).")

    args = parser.parse_args()
    if args.config_file:
        config = load_config(args.config_file)
        for option, value in config.items():
            setattr(args, option, value)
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


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
