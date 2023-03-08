import os
import torch
import itertools
from tqdm import tqdm
from einops import rearrange, repeat
from torch import nn


def get_item_embedding(
    batch_y,
    model,
    mode
):
    item, item_feats = batch_y
    x = model.id_embedding(item).unsqueeze(1)
    if mode == "not":
        return x.squeeze(1)
    
    item_feat = []
    item_feats.append(item)
    for item_feat_list, feat_embedding in zip(item_feats, model.feat_embeddings):
        item_feat.append(feat_embedding(item_feat_list).unsqueeze(1))
    item_feat = torch.cat(item_feat, axis=1)

    if mode == "attn":
        x = model.cross_attn(x, context=item_feat) + x
        x = model.cross_ff(x) + x
    elif mode == "mean":
        x = torch.cat([x, item_feat], axis=1).mean(axis=1, keepdim=True)        
    return x.squeeze(1)


def train(
    start_epoch,
    num_epochs,
    early_stop,
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    writer,
    logger,
    log_dir,
    device,
    item_fusion_mode,
    one_to_one_loss,
):
    model.train()
    best_val = 0.0
    best_save_dir = os.path.join(log_dir, "best")
    patience = 0
    logger.info(f"Training starts - {num_epochs} epochs")
    if criterion == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = ContrastiveLoss(train_loader.batch_size)
    for e in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        train_loader = tqdm(train_loader)
        scheduler.step()
        for batch_x, batch_y_pos, batch_y_negs in train_loader:
            optimizer.zero_grad()
            x_out_list = model(batch_x)
            if not one_to_one_loss:
                x_out_list = x_out_list[-1:]
            y_pos_out = get_item_embedding(batch_y_pos, model, item_fusion_mode)
            loss = 0
            for x_out in x_out_list:
                if criterion == "BCE":
                    items, i_feats = batch_y_negs
                    batch_size, n_negs = items.shape
                    items = rearrange(items, "b n -> (b n)")
                    i_feats = [rearrange(i_feat, "b n d -> (b n) d") for i_feat in i_feats]
                    y_negs_out = get_item_embedding((items, i_feats), model, item_fusion_mode)
                    x_out_extended = rearrange(repeat(x_out.unsqueeze(1), "b 1 d -> b n d", n=n_negs), "b n d -> (b n) d")
                    neg_score = torch.diag(x_out_extended @ y_negs_out.T)
                    neg_score = rearrange(neg_score, "(b n) -> b n", b=batch_size)
                    neg_label = torch.zeros(neg_score.shape).to(device).float()

                    pos_score = torch.diag(x_out @ y_pos_out.T)
                    pos_score = rearrange(pos_score, "(b n) -> b n", b=batch_size)
                    pos_label = torch.ones(pos_score.shape).to(device).float()

                    logits = torch.cat([pos_score, neg_score], axis=1)
                    label = torch.cat([pos_label, neg_label], axis=1)
                    loss += loss_fn(logits, label)
                else:
                    loss += loss_fn(x_out, y_pos_out)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            train_loader.set_description(f"Epoch {e} - loss: {loss.item():.4f}")
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {e} - lr: {current_lr}, loss: {epoch_loss}")
        writer.add_scalar("Loss/train", epoch_loss, e)
        writer.add_scalar("LR", current_lr, e)
        val_metrics = evaluate(model, val_loader, item_fusion_mode, criterion=criterion, loss_fn=loss_fn)
        val_log = ""
        for k, v in val_metrics.items():
            writer.add_scalar(f"{k}/valid", v, e)
            val_log += f"{k}: {v:.5f} "
        logger.info(f"Epoch {e} validation - {val_log}")
        save_dict = {
            "last_epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        val = val_metrics["NDCG@10"]
        if val > best_val:
            best_val = val
            patience = 0
            logger.info("Save best model")
            save_model(best_save_dir, save_dict)
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stopping at epoch {e}")
                break
        save_model(log_dir, save_dict)

    best_model_path = os.path.join(best_save_dir, "model.pt")
    model.load_state_dict(torch.load(best_model_path))
    return model


def evaluate(
    model,
    dataloader,
    item_fusion_mode,
    criterion=None,
    loss_fn=None,
):
    model.eval()
    num_users = len(dataloader.dataset)
    dataloader = tqdm(dataloader)
    metrics_handler = MetricsHandler(num_users=num_users)
    with torch.no_grad():
        for batch_x, batch_y_pos, batch_y_negs in dataloader:
            x_out = model(batch_x)[-1]
            y_pos_out = get_item_embedding(batch_y_pos, model, item_fusion_mode)
            items, i_feats = batch_y_negs
            batch_size, n_negs = items.shape
            items = rearrange(items, "b n -> (b n)")
            i_feats = [rearrange(i_feat, "b n d -> (b n) d") for i_feat in i_feats]
            y_negs_out = get_item_embedding((items, i_feats), model, item_fusion_mode)

            pos_score = torch.diag(x_out @ y_pos_out.T)
            pos_score = rearrange(pos_score, "(b n) -> b n", b=batch_size)
            x_out_extended = rearrange(repeat(x_out.unsqueeze(1), "b 1 d -> b n d", n=n_negs), "b n d -> (b n) d")
            neg_score = torch.diag(x_out_extended @ y_negs_out.T)
            neg_score = rearrange(neg_score, "(b n) -> b n", b=batch_size)
            scores = torch.cat([pos_score, neg_score], axis=1)
            scores = scores.detach().cpu()
            labels = torch.cat([torch.ones(pos_score.shape), torch.zeros(neg_score.shape)], axis=1)
            metrics = calculate_metrics(scores, labels, k_list=[1, 5, 10])
            if criterion == "BCE":
                loss = loss_fn(scores, labels)
                metrics["Loss"] = loss.item()
            elif criterion == "CL":
                loss = loss_fn(x_out, y_pos_out)
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
            if milestones and gamma and str(step) in milestones:
                self.gamma *= gamma
            return self.gamma

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temp=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temp = temp
        self.neg_mask = self.get_neg_mask(batch_size)
        self.sim_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def get_neg_mask(self, batch_size):
        neg_mask = torch.ones((batch_size, batch_size), dtype=bool)
        neg_mask.fill_diagonal_(0)
        return neg_mask
    
    def forward(self, seq_out, item_out):
        batch_size = seq_out.shape[0]
        sim = self.sim_f(seq_out.unsqueeze(1), item_out.unsqueeze(0)) / self.temp
        pos_samples = sim.diag().reshape(batch_size, -1)
        neg_mask = self.neg_mask if batch_size == self.batch_size else self.get_neg_mask(batch_size)
        neg_samples = sim[neg_mask].reshape(batch_size, -1)

        labels = torch.zeros(batch_size).to(pos_samples.device).long()
        logits = torch.cat((pos_samples, neg_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss / batch_size