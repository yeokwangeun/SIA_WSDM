import os
import torch
import itertools
from tqdm import tqdm
import torch.nn.functional as F


def train(
    start_epoch,
    num_epochs,
    early_stop,
    train_loader,
    val_loader,
    eval_sample_mode,
    num_items,
    model,
    optimizer,
    scheduler,
    loss_fn,
    writer,
    logger,
    log_dir
):
    model.train()
    best_val = 0.0
    patience = 0
    logger.info(f"Training starts - {num_epochs} epochs")
    for e in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        train_loader = tqdm(train_loader)
        scheduler.step()
        for seq_list, pos_list, next_item_list, *item_feat_lists in train_loader:
            optimizer.zero_grad()
            x = (seq_list, pos_list, *item_feat_lists)
            logits = model(x)
            loss = loss_fn(logits, next_item_list)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            train_loader.set_description(f"Epoch {e} - loss: {loss.item():.4f}")
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {e} - lr: {current_lr}, loss: {epoch_loss}")
        writer.add_scalar("Loss/train", epoch_loss, e)
        writer.add_scalar("LR", current_lr, e)
        val_metrics = evaluate(model, val_loader, eval_sample_mode, num_items, loss_fn=loss_fn)
        val_log = ""
        for k, v in val_metrics.items():
            writer.add_scalar(f"{k}/valid", v, e)
            val_log += f"{k}: {v:.5f} "
        logger.info(f"Epoch {e} validation - {val_log}")
        val = val_metrics["NDCG@10"]
        if val > best_val:
            best_val = val
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stopping at epoch {e}")
                break
        save_dict = {
            'last_epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        save_model(log_dir, save_dict)
    writer.flush()
    writer.close()
    return (model, e)


def evaluate(
    model,
    dataloader,
    sample_mode,
    num_items,
    loss_fn=None,
):
    model.eval()
    num_users = len(dataloader.dataset)
    dataloader = tqdm(dataloader)
    metrics_handler = MetricsHandler(num_users=num_users)
    with torch.no_grad():
        for seq_list, pos_list, next_item_list, candidate_list, *item_feat_lists in dataloader:
            x = (seq_list, pos_list, *item_feat_lists)
            logits = model(x)
            scores = logits if sample_mode == "full" else logits.gather(1, candidate_list)
            scores = scores.detach().cpu()
            labels = F.one_hot(next_item_list.detach().cpu(), num_classes=(num_items + 1))
            labels = (
                labels if sample_mode == "full" else labels.gather(1, candidate_list.detach().cpu())
            )
            metrics = calculate_metrics(scores, labels, k_list=[1, 5, 10])
            if loss_fn:
                loss = loss_fn(logits, next_item_list)
                metrics["Loss"] = loss.item()
            metrics_handler.append_metrics(metrics)
    return metrics_handler.get_metrics()


def calculate_metrics(scores, labels, k_list=[1, 5, 10]):
    metrics = [f"{metric}@{k}" for metric, k in itertools.product(["NDCG", "HR"], k_list)]
    metrics_sum = {metric: 0.0 for metric in metrics}
    rank = (-scores).argsort(dim=1)
    labels_float = labels.float()
    for k in k_list:
        cut = rank[:, :k]
        hits = labels_float.gather(dim=1, index=cut)
        metrics_sum[f"HR@{k}"] = hits.sum().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = hits * weights
        metrics_sum[f"NDCG@{k}"] = dcg.sum().item()

    return metrics_sum


def save_model(save_dir, save_dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, "model.pt")
    chkpoint_save_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(save_dict["model_state_dict"], model_save_path)
    torch.save(save_dict, chkpoint_save_path)


class MetricsHandler:
    def __init__(self, num_users):
        self.metrics = None
        self.num_users = num_users

    def append_metrics(self, metrics):
        if not self.metrics:
            self.metrics = metrics
        else:
            for k, v in metrics.items():
                self.metrics[k] += v

    def get_metrics(self):
        return {k: ((v / self.num_users) if k != "Loss" else v) for k, v in self.metrics.items()}


class WarmupBeforeMultiStepLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_step=None, milestones=None, gamma=None, last_epoch=-1):
        self.gamma = 1

        def lr_lambda(step):
            if warmup_step and step < warmup_step:
                return step / warmup_step
            if milestones and gamma and step in milestones:
                self.gamma *= gamma
            return self.gamma

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
