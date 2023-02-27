import os
import torch
import itertools
from tqdm import tqdm
from einops import rearrange, repeat
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
    item_model,
    seq_optimizer,
    item_optimizer,
    seq_scheduler,
    item_scheduler,
    loss_fn,
    writer,
    logger,
    log_dir,
    device
):
    model.train()
    item_model.train()
    best_val = 0.0
    patience = 0
    logger.info(f"Training starts - {num_epochs} epochs")
    for e in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        train_loader = tqdm(train_loader)
        seq_scheduler.step()
        item_scheduler.step()
        for batch_x, batch_y_pos, batch_y_neg in train_loader:
            seq_optimizer.zero_grad()
            item_optimizer.zero_grad()
            x_out = model(batch_x)
            y_pos_out = item_model(batch_y_pos)
            y_neg_out = item_model(batch_y_neg)
            pos_score = torch.diag(x_out @ y_pos_out.T)
            pos_label = torch.ones(pos_score.shape).to(device)
            neg_score = torch.diag(x_out @ y_neg_out.T)
            neg_label = torch.zeros(neg_score.shape).to(device)
            loss = loss_fn(pos_score, pos_label) + loss_fn(neg_score, neg_label)
            loss.backward()
            epoch_loss += loss.item()
            seq_optimizer.step()
            item_optimizer.step()
            train_loader.set_description(f"Epoch {e} - loss: {loss.item():.4f}")
        current_lr = seq_optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {e} - lr: {current_lr}, loss: {epoch_loss}")
        writer.add_scalar("Loss/train", epoch_loss, e)
        writer.add_scalar("LR", current_lr, e)
        val_metrics = evaluate(model, item_model, val_loader, eval_sample_mode, num_items, loss_fn=loss_fn)
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
            "last_epoch": e,
            "model_state_dict": model.state_dict(),
            "seq_optimizer_state_dict": seq_optimizer.state_dict(),
            "item_optimizer_state_dict": item_optimizer.state_dict(),
            "seq_scheduler_state_dict": seq_scheduler.state_dict(),
            "item_scheduler_state_dict": item_scheduler.state_dict(),
        }
        save_model(log_dir, save_dict)
    writer.flush()
    writer.close()
    return model


def evaluate(
    model,
    item_model,
    dataloader,
    sample_mode,
    num_items,
    loss_fn=None,
):
    model.eval()
    item_model.eval()
    num_users = len(dataloader.dataset)
    dataloader = tqdm(dataloader)
    metrics_handler = MetricsHandler(num_users=num_users)
    with torch.no_grad():
        for batch_x, batch_y_pos, batch_y_negs in dataloader:
            x_out = model(batch_x)
            y_pos_out = item_model(batch_y_pos)
            items, i_feats = batch_y_negs
            batch_size, n_negs = items.shape
            items = rearrange(items, "b n -> (b n)")
            i_feats = [rearrange(i_feat, "b n d -> (b n) d") for i_feat in i_feats]
            y_negs_out = item_model((items, i_feats))

            pos_score = torch.diag(x_out @ y_pos_out.T)
            pos_score = rearrange(pos_score, "(b n) -> b n", b=batch_size)
            x_out_extended = rearrange(repeat(x_out.unsqueeze(1), "b 1 d -> b n d", n=n_negs), "b n d -> (b n) d")
            neg_score = torch.diag(x_out_extended @ y_negs_out.T)
            neg_score = rearrange(neg_score, "(b n) -> b n", b=batch_size)
            scores = torch.cat([pos_score, neg_score], axis=1)
            scores = scores.detach().cpu()
            labels = torch.cat([torch.ones(pos_score.shape), torch.zeros(neg_score.shape)], axis=1)
            metrics = calculate_metrics(scores, labels, k_list=[1, 5, 10])
            if loss_fn:
                loss = loss_fn(scores, labels)
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
