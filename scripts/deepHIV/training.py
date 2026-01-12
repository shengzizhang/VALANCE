import os
import time

from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import numpy as np
import pandas as pd
import torch as t
from deep_hiv_ab_pred.util.metrics import compute_metrics
import optuna
from deep_hiv_ab_pred.util.tools import to_numpy
import logging
from deep_hiv_ab_pred.training.cv_pruner import CrossValidationPruner
import matplotlib.pyplot as plt
# from labml_nn.optimizers.noam import Noam
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F



class FocalLoss(t.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        # Ensure the weight tensor is on the same device as input_tensor
        if self.weight is not None:
            self.weight = self.weight.to(input_tensor.device)

        # Calculate log probabilities
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = t.exp(log_prob)

        # Calculate the focal weight
        focal_weight = (1 - prob) ** self.gamma

        # Compute the loss with NLL and class weights
        loss = F.nll_loss(focal_weight * log_prob, target_tensor, weight=self.weight, reduction=self.reduction)
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2): 
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#     def forward(self, inputs, targets): 
#         BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#         return torch.mean(F_loss)

class WeightedFocalLoss(t.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = t.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, input_tensor, target_tensor):
        BCE_loss = F.binary_cross_entropy_with_logits(input_tensor, target_tensor, reduction='none')
        target_tensor = target_tensor.type(t.long)
        at = self.alpha.gather(0, target_tensor.data.view(-1))
        pt = t.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def run_network_for_training(model, conf, loader, loss_fn, optimizer, epochs = None, pruner = None):
    metrics = np.zeros(3)
    # we calculate a weighted average by the number of samples in each batch,
    # all batches will have the same number of elements (weight one), except
    # for the last one which will have less elements (will have subunitary weight)
    total_weight = 0
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        start = time.time()
        pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
        if pred.shape != ground_truth.shape:
            pred = pred.reshape(ground_truth.shape)
        loss = loss_fn(pred, ground_truth)
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
        optimizer.step()
        optimizer.zero_grad()
        # The last batch have fewer elements then the rest.
        # For this reason we weight each metric by the population size of the batch using the variable named 'weight'
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(to_numpy(ground_truth), to_numpy(pred)) * weight
        if i > 1 and i < 10 and epochs and pruner:
            estimated_time = epochs * len(loader) * (time.time() - start) / 60
            pruner.report_time(estimated_time)
    return metrics / total_weight


# def run_net_with_frozen_antibody_and_embedding(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
#     metrics = np.zeros(3)
#     total_weight = 0
#     for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
#         batch_size = len(ab_light)
#         with t.no_grad():
#             ab_light, ab_heavy, virus = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
#             ab_hidden = model.module.forward_antibodyes(ab_light, ab_heavy)
#         pred = model.module.forward_virus(virus, pngs_mask, ab_hidden)
#         if pred.shape != ground_truth.shape:
#             pred = pred.reshape(ground_truth.shape)
#         loss = loss_fn(pred, ground_truth)
#         if isTrain:
#             assert optimizer != None
#             loss.backward()
#             t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
#             optimizer.step()
#             optimizer.zero_grad()
#         weight = len(ground_truth) / conf['BATCH_SIZE']
#         total_weight += weight
#         metrics += compute_metrics(to_numpy(ground_truth), to_numpy(pred)) * weight
#     return metrics / total_weight

def run_net_with_frozen_antibody_and_embedding(model, conf, loader, loss_fn, optimizer=None, isTrain=False):
    metrics = np.zeros(3)
    total_weight = 0
    total_loss = 0  # Added to track total loss
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        batch_size = len(ab_light)
        with t.no_grad():
            ab_light, ab_heavy, virus = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
            ab_hidden = model.module.forward_antibodyes(ab_light, ab_heavy)
        pred = model.module.forward_virus(virus, pngs_mask, ab_hidden)
        if pred.shape != ground_truth.shape:
            pred = pred.reshape(ground_truth.shape)
        loss = loss_fn(pred, ground_truth)
        total_loss += loss.item() * batch_size  # Track total loss
        if isTrain:
            assert optimizer is not None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(to_numpy(ground_truth), to_numpy(pred)) * weight
    avg_loss = total_loss / total_weight  # Compute average loss
    return metrics / total_weight, avg_loss  # Return average loss

def run_net_with_frozen_antibody_and_embedding_GRU(model, conf, loader, loss_fn, optimizer = None, isTrain = False):
    metrics = np.zeros(3)
    total_weight = 0
    for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
        batch_size = len(ab_light)
        with t.no_grad():
            ab_light, ab_heavy, virus = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
            ab_hidden = model.module.forward_antibodyes(ab_light, ab_heavy)
        pred = model.module.forward_virus(virus, pngs_mask, ab_hidden)
        if pred.shape != ground_truth.shape:
            pred = pred.reshape(ground_truth.shape)
        loss = loss_fn(pred, ground_truth)
        if isTrain:
            assert optimizer != None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()
        weight = len(ground_truth) / conf['BATCH_SIZE']
        total_weight += weight
        metrics += compute_metrics(to_numpy(ground_truth), to_numpy(pred)) * weight
    return metrics / total_weight

def eval_network_cv(model, conf, loader, loss_fn):
    model.eval()
    prediction_list, ground_truth_list = [], []
    metrics = np.zeros(3)
    total_weight = 0
    total_loss = 0  # Added to track total loss
    with t.no_grad():
        for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
            batch_size = len(ab_light)
            pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            if pred.shape != ground_truth.shape:
                pred = pred.reshape(ground_truth.shape)
            prediction_list.append(to_numpy(pred))
            ground_truth_list.append(to_numpy(ground_truth))
            loss = loss_fn(pred, ground_truth)
            total_loss += loss.item() * batch_size  # Track total loss
            weight = len(ground_truth) / conf['BATCH_SIZE']
            total_weight += weight
    all_predictions = np.concatenate(prediction_list)
    all_ground_truths = np.concatenate(ground_truth_list)
    avg_loss = total_loss / total_weight  # Compute average loss
    return compute_metrics(all_ground_truths, all_predictions, include_AUC = True), avg_loss  # Return average loss

# Modifying eval_network_cv to return not only metrics and loss but also the full lists of predictions and ground‐truth values.
def eval_network_cv_pred(model, conf, loader, loss_fn):
    model.eval()
    prediction_list, ground_truth_list = [], []
    total_weight = 0
    total_loss = 0  # Track total loss
    with t.no_grad():
        for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
            batch_size = len(ab_light)
            pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            if pred.shape != ground_truth.shape:
                pred = pred.reshape(ground_truth.shape)
            prediction_list.append(to_numpy(pred))
            ground_truth_list.append(to_numpy(ground_truth))
            loss = loss_fn(pred, ground_truth)
            total_loss += loss.item() * batch_size
            weight = len(ground_truth) / conf['BATCH_SIZE']
            total_weight += weight
    all_predictions = np.concatenate(prediction_list)
    all_ground_truths = np.concatenate(ground_truth_list)
    avg_loss = total_loss / total_weight  # Compute average loss
    metrics = compute_metrics(all_ground_truths, all_predictions, include_AUC=True)
    # Return metrics, average loss, and the full predictions/ground truths
    return metrics, avg_loss, all_predictions, all_ground_truths


def eval_network_cv_pred_confidence(model, conf, loader, loss_fn, conf_thresh=0.60):
    """
    Evaluate a model and return:
        metrics      – MCC, AUC, ACC on the *kept* samples
        avg_loss     – average loss over *all* samples
        preds_keep   – numpy array of kept probabilities
        gts_keep     – numpy array of kept ground‑truth labels
    
    Additionally prints the number of discarded instances and a list of them.
    """
    model.eval()

    preds_all, gts_all = [], []
    sample_idx_all     = []        # to track which global index each pred belongs to
    total_loss, total_weight = 0.0, 0.0
    running_idx = 0                # global counter across the loader

    with t.no_grad():
        for ab_light, ab_heavy, virus, pngs_mask, y in loader:
            batch_size = len(y)

            p = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            if p.shape != y.shape:          # safety reshape
                p = p.reshape(y.shape)

            preds_all.append(p.cpu().numpy())
            gts_all.append(y.cpu().numpy())
            sample_idx_all.extend(range(running_idx, running_idx + batch_size))
            running_idx += batch_size

            total_loss   += loss_fn(p, y).item() * batch_size
            total_weight += batch_size

    # flatten lists
    preds_all = np.concatenate(preds_all)
    gts_all   = np.concatenate(gts_all)
    sample_idx_all = np.array(sample_idx_all)

    # confidence & masks
    conf_scores = np.maximum(preds_all, 1.0 - preds_all)
    keep_mask   = conf_scores >= conf_thresh
    discard_mask = ~keep_mask

    preds_keep = preds_all[keep_mask]
    gts_keep   = gts_all[keep_mask]

    # --------- print discarded instances ------------------------------------
    disc_indices  = sample_idx_all[discard_mask]
    disc_probs    = preds_all[discard_mask]
    disc_confs    = conf_scores[discard_mask]
    disc_labels   = gts_all[discard_mask]

    print(f"\nDiscarded {len(disc_indices)} instances (confidence < {conf_thresh}):")
    for idx, prob, confc, gt in zip(disc_indices, disc_probs, disc_confs, disc_labels):
        print(f"  idx={idx:>5}  prob={prob:.4f}  conf={confc:.4f}  label={int(gt)}")
    print("-" * 60)

    # metrics only on kept instances
    metrics = compute_metrics(gts_keep, preds_keep, include_AUC=True)
    avg_loss = total_loss / total_weight

    return metrics, avg_loss, preds_keep, gts_keep


    
def eval_network(model, loader):
    model.eval()
    prediction_list, ground_truth_list = [], []
    with t.no_grad():
        for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
            pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            if pred.shape != ground_truth.shape:
                pred = pred.reshape(ground_truth.shape)
            prediction_list.append(to_numpy(pred))
            ground_truth_list.append(to_numpy(ground_truth))
    all_predictions = np.concatenate(prediction_list)
    all_ground_truths = np.concatenate(ground_truth_list)
    return compute_metrics(all_ground_truths, all_predictions, include_AUC = True)

# def eval_network_pred(model, loader):
#     model.eval()
#     prediction_list, ground_truth_list = ['Holaaaaaaaaaaa'], []
#     print(prediction_list)
#     with t.no_grad():
#         for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
#             pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
#             prediction_list.append(to_numpy(pred))
            
#             ground_truth_list.append(to_numpy(ground_truth))
#     all_predictions = np.concatenate(prediction_list)
#     all_ground_truths = np.concatenate(ground_truth_list)
#     return all_predictions, all_ground_truths


def eval_network_pred(model, loader):
    model.eval()
    prediction_list, ground_truth_list = [], [] 
    with t.no_grad():
        for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
            # Forward pass
            pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            
            # Convert predictions and ground truth to numpy arrays
            pred_np = to_numpy(pred).flatten()  # Ensure it's a 1D array
            ground_truth_np = to_numpy(ground_truth).flatten()  # Ensure it's a 1D array
            
            # Append to lists
            prediction_list.append(pred_np)
            ground_truth_list.append(ground_truth_np)
            
            # # Debugging: Print shapes
            # print(f"Batch {i}: pred shape = {pred_np.shape}, ground truth shape = {ground_truth_np.shape}")

    # Concatenate all predictions and ground truths
    all_predictions = np.concatenate(prediction_list)
    all_ground_truths = np.concatenate(ground_truth_list)
    
    return all_predictions, all_ground_truths

# def eval_network(model, loader):
#     model.eval()
#     prediction_list, ground_truth_list = [], []
#     val_loss = 0.0
#     loss_fn = t.nn.BCELoss()
#     with t.no_grad():
#         for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
#             pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
#             if pred.shape != ground_truth.shape:
#                 pred = pred.reshape(ground_truth.shape)
#             prediction_list.append(to_numpy(pred))
#             ground_truth_list.append(to_numpy(ground_truth))
#             val_loss += loss_fn(pred, ground_truth).item()

#     all_predictions = np.concatenate(prediction_list)
#     all_ground_truths = np.concatenate(ground_truth_list)
#     metrics = compute_metrics(all_ground_truths, all_predictions, include_AUC=True)
#     val_loss /= len(loader)
#     return metrics, val_loss

# def eval_network(model, loader):
#     model.eval()
#     prediction_list, ground_truth_list = [], []
#     val_loss = 0.0
#     total_samples = 0
#     loss_fn = t.nn.BCELoss()
#     with t.no_grad():
#         for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
#             pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
#             if pred.shape != ground_truth.shape:
#                 pred = pred.reshape(ground_truth.shape)
#             loss = loss_fn(pred, ground_truth)
#             val_loss += loss.item() * len(ground_truth)
#             total_samples += len(ground_truth)
#             prediction_list.append(to_numpy(pred))
#             ground_truth_list.append(to_numpy(ground_truth))

#     all_predictions = np.concatenate(prediction_list)
#     all_ground_truths = np.concatenate(ground_truth_list)
#     metrics = compute_metrics(all_ground_truths, all_predictions, include_AUC=True)
#     val_loss /= total_samples  # Compute the mean validation loss
#     return metrics, val_loss
    
# def eval_network_cv(model, loader, loss_fn):
#     model.eval()
#     prediction_list, ground_truth_list = [], []
#     total_loss = 0
#     total_samples = 0
#     with t.no_grad():
#         for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
#             pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
#             if pred.shape != ground_truth.shape:
#                 pred = pred.reshape(ground_truth.shape)
#             loss = loss_fn(pred, ground_truth)
#             total_loss += loss.item() * len(ground_truth)
#             total_samples += len(ground_truth)
#             prediction_list.append(to_numpy(pred))
#             ground_truth_list.append(to_numpy(ground_truth))
#     all_predictions = np.concatenate(prediction_list)
#     all_ground_truths = np.concatenate(ground_truth_list)
#     metrics = compute_metrics(all_ground_truths, all_predictions, include_AUC=True)
#     val_loss = total_loss / total_samples  # Calculate average loss
#     return metrics, val_loss


    

# Train
def train_network(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = np.zeros(3)
    try:
        for epoch in range(epochs):
            model.train()
            train_metrics = run_network_for_training(model, conf, loader_train, loss_fn, optimizer)
            metrics_train_per_epochs.append(train_metrics)
            if loader_val:
                test_metrics = eval_network(model, loader_val)
                metrics_test_per_epochs.append(test_metrics)
                # We save a model chekpoint if we find any improvement
                if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = test_metrics
                    if save_model:
                        if cross_validation_round is not None:
                            t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
                        else:
                            t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
                if log_every_epoch:
                    logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                # We save a model chekpoint if we find any improvement
                if train_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                    best = train_metrics
                    if save_model:
                        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
                if log_every_epoch:
                    logging.info(f'Epoch {epoch + 1}, Correlation: {train_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {train_metrics[ACCURACY]}')
        if cross_validation_round is not None:
            logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        else:
            logging.info(f'Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)



# def train_with_frozen_antibody_and_embedding(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
#     # Freezing the embeddings and antibody subnetworks
#     for param in model.module.aminoacid_embedding.parameters():
#         param.requires_grad = False
#     model.module.embedding_dropout = t.nn.Dropout(p = 0)
#     model.module.embedding_dropout.requires_grad = False

#     for param in model.module.light_ab_fc.parameters():
#         param.requires_grad = False
#     model.module.light_ab_dropout.dropout = 0
#     model.module.light_ab_dropout.requires_grad = False

#     for param in model.module.light_ab_att.parameters():
#         param.requires_grad = False
#     model.module.light_ab_att_dropout.dropout = 0
#     model.module.light_ab_att_dropout.requires_grad = False

#     for param in model.module.heavy_ab_fc.parameters():
#         param.requires_grad = False
#     model.module.heavy_ab_dropout.dropout = 0
#     model.module.heavy_ab_dropout.requires_grad = False

#     for param in model.module.heavy_ab_att.parameters():
#         param.requires_grad = False
#     model.module.heavy_ab_att_dropout.dropout = 0
#     model.module.heavy_ab_att_dropout.requires_grad = False

#     loss_fn = t.nn.BCELoss()
#     optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
#     metrics_train_per_epochs, metrics_test_per_epochs = [], []
#     best = np.zeros(3)
#     try:
#         for epoch in range(epochs):
#             model.module.aminoacid_embedding.eval()
#             model.module.embedding_dropout.eval()
#             model.module.light_ab_fc.eval()
#             model.module.light_ab_dropout.eval()
#             model.module.light_ab_att.eval()
#             model.module.light_ab_att_dropout.eval()
#             model.module.heavy_ab_fc.eval()
#             model.module.heavy_ab_dropout.eval()
#             model.module.heavy_ab_att.eval()
#             model.module.heavy_ab_att_dropout.eval()

#             model.module.virus_gru.train()
#             model.module.fc_dropout.train()
#             model.module.fully_connected.train()

#             train_metrics = run_net_with_frozen_antibody_and_embedding(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
#             metrics_train_per_epochs.append(train_metrics)

#             test_metrics = eval_network(model, loader_val)
#             metrics_test_per_epochs.append(test_metrics)
#             # We save a model chekpoint if we find any improvement
#             if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
#                 best = test_metrics
#                 if save_model:
#                     t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
#             if log_every_epoch:
#                 logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

#         logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
#         return metrics_train_per_epochs, metrics_test_per_epochs, best
#     except KeyboardInterrupt as e:
#         logging.info('Training interrupted at epoch ' + epoch)

def train_with_frozen_antibody_and_embedding(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title='model', model_path='', save_model=True, log_every_epoch=True):
    # Freezing the embeddings and antibody subnetworks
    for param in model.module.aminoacid_embedding.parameters():
        param.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p=0)
    model.module.embedding_dropout.requires_grad = False

    for param in model.module.light_ab_fc.parameters():
        param.requires_grad = False
    model.module.light_ab_dropout.dropout = 0
    model.module.light_ab_dropout.requires_grad = False

    for param in model.module.light_ab_att.parameters():
        param.requires_grad = False
    model.module.light_ab_att_dropout.dropout = 0
    model.module.light_ab_att_dropout.requires_grad = False

    for param in model.module.heavy_ab_fc.parameters():
        param.requires_grad = False
    model.module.heavy_ab_dropout.dropout = 0
    model.module.heavy_ab_dropout.requires_grad = False

    for param in model.module.heavy_ab_att.parameters():
        param.requires_grad = False
    model.module.heavy_ab_att_dropout.dropout = 0
    model.module.heavy_ab_att_dropout.requires_grad = False

    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    train_mcc, val_mcc = [], []
    train_losses, val_losses = [], []
    best = np.zeros(3)
    try:
        for epoch in range(epochs):
            model.module.aminoacid_embedding.eval()
            model.module.embedding_dropout.eval()
            model.module.light_ab_fc.eval()
            model.module.light_ab_dropout.eval()
            model.module.light_ab_att.eval()
            model.module.light_ab_att_dropout.eval()
            model.module.heavy_ab_fc.eval()
            model.module.heavy_ab_dropout.eval()
            model.module.heavy_ab_att.eval()
            model.module.heavy_ab_att_dropout.eval()

            model.module.virus_gru.train()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics, train_loss = run_net_with_frozen_antibody_and_embedding(model, conf, loader_train, loss_fn, optimizer, isTrain=True)
            metrics_train_per_epochs.append(train_metrics)
            train_losses.append(train_loss)
            train_mcc.append(train_metrics[MATTHEWS_CORRELATION_COEFFICIENT])

            # model.module.virus_gru.eval()
            # model.module.fc_dropout.eval()
            # model.module.fully_connected.eval()

            # test_metrics, val_loss = eval_network_cv(model, loader_val, loss_fn)
            test_metrics, val_loss = eval_network_cv(model, conf, loader_val, loss_fn)
            metrics_test_per_epochs.append(test_metrics)
            val_mcc.append(test_metrics[MATTHEWS_CORRELATION_COEFFICIENT])
            val_losses.append(val_loss)

            # print(f"Epoch {epoch + 1} - Train Loss: {train_loss}, Validation Loss: {val_loss}")
            
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
            if log_every_epoch:
                logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        

        return metrics_train_per_epochs, metrics_test_per_epochs, best, train_mcc, val_mcc, train_losses, val_losses
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)

def train_with_frozen_antibody_and_embedding_predict(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title='model', model_path='', save_model=True, log_every_epoch=True):
    # Freezing the embeddings and antibody subnetworks
    for param in model.module.aminoacid_embedding.parameters():
        param.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p=0)
    model.module.embedding_dropout.requires_grad = False

    for param in model.module.light_ab_fc.parameters():
        param.requires_grad = False
    model.module.light_ab_dropout.dropout = 0
    model.module.light_ab_dropout.requires_grad = False

    for param in model.module.light_ab_att.parameters():
        param.requires_grad = False
    model.module.light_ab_att_dropout.dropout = 0
    model.module.light_ab_att_dropout.requires_grad = False

    for param in model.module.heavy_ab_fc.parameters():
        param.requires_grad = False
    model.module.heavy_ab_dropout.dropout = 0
    model.module.heavy_ab_dropout.requires_grad = False

    for param in model.module.heavy_ab_att.parameters():
        param.requires_grad = False
    model.module.heavy_ab_att_dropout.dropout = 0
    model.module.heavy_ab_att_dropout.requires_grad = False

    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    train_mcc, val_mcc = [], []
    train_losses, val_losses = [], []
    best = np.zeros(3)
    try:
        for epoch in range(epochs):
            model.module.aminoacid_embedding.eval()
            model.module.embedding_dropout.eval()
            model.module.light_ab_fc.eval()
            model.module.light_ab_dropout.eval()
            model.module.light_ab_att.eval()
            model.module.light_ab_att_dropout.eval()
            model.module.heavy_ab_fc.eval()
            model.module.heavy_ab_dropout.eval()
            model.module.heavy_ab_att.eval()
            model.module.heavy_ab_att_dropout.eval()

            model.module.virus_gru.train()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics, train_loss = run_net_with_frozen_antibody_and_embedding(model, conf, loader_train, loss_fn, optimizer, isTrain=True)
            metrics_train_per_epochs.append(train_metrics)
            train_losses.append(train_loss)
            train_mcc.append(train_metrics[MATTHEWS_CORRELATION_COEFFICIENT])

            # model.module.virus_gru.eval()
            # model.module.fc_dropout.eval()
            # model.module.fully_connected.eval()

            # test_metrics, val_loss = eval_network_cv(model, loader_val, loss_fn)
            # test_metrics, val_loss, all_predictions, all_ground_truths = eval_network_cv_pred(model, conf, loader_val, loss_fn)
            test_metrics, val_loss, all_predictions, all_ground_truths = eval_network_cv_pred_confidence(model, conf, loader_val, loss_fn, 0.6)
            metrics_test_per_epochs.append(test_metrics)
            val_mcc.append(test_metrics[MATTHEWS_CORRELATION_COEFFICIENT])
            val_losses.append(val_loss)

            # print(f"Epoch {epoch + 1} - Train Loss: {train_loss}, Validation Loss: {val_loss}")
            
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
            if log_every_epoch:
                logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        

        return metrics_train_per_epochs, metrics_test_per_epochs, best, train_mcc, val_mcc, train_losses, val_losses, all_predictions, all_ground_truths
        
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)

def train_with_frozen_antibody_and_embedding_focal_loss(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title='model', model_path='', save_model=True, log_every_epoch=True):
    # Freezing the embeddings and antibody subnetworks
    for param in model.module.aminoacid_embedding.parameters():
        param.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p=0)
    model.module.embedding_dropout.requires_grad = False

    for param in model.module.light_ab_fc.parameters():
        param.requires_grad = False
    model.module.light_ab_dropout.dropout = 0
    model.module.light_ab_dropout.requires_grad = False

    for param in model.module.light_ab_att.parameters():
        param.requires_grad = False
    model.module.light_ab_att_dropout.dropout = 0
    model.module.light_ab_att_dropout.requires_grad = False

    for param in model.module.heavy_ab_fc.parameters():
        param.requires_grad = False
    model.module.heavy_ab_dropout.dropout = 0
    model.module.heavy_ab_dropout.requires_grad = False

    for param in model.module.heavy_ab_att.parameters():
        param.requires_grad = False
    model.module.heavy_ab_att_dropout.dropout = 0
    model.module.heavy_ab_att_dropout.requires_grad = False

    # # Compute class weights based on training labels
    # train_labels = [int(a[3]) for a in loader_train.dataset.assays]
    # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=np.array(train_labels))
    # device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # class_weights = t.tensor(class_weights, dtype=t.float).to(device)  # Move to GPU

    # Use FocalLoss with class weights
    # loss_fn = FocalLoss(weight=class_weights, gamma=2.0, reduction='none')

    loss_fn = WeightedFocalLoss(alpha=.25, gamma=2)
    
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    train_mcc, val_mcc = [], []
    train_losses, val_losses = [], []
    best = np.zeros(3)
    try:
        for epoch in range(epochs):
            model.module.aminoacid_embedding.eval()
            model.module.embedding_dropout.eval()
            model.module.light_ab_fc.eval()
            model.module.light_ab_dropout.eval()
            model.module.light_ab_att.eval()
            model.module.light_ab_att_dropout.eval()
            model.module.heavy_ab_fc.eval()
            model.module.heavy_ab_dropout.eval()
            model.module.heavy_ab_att.eval()
            model.module.heavy_ab_att_dropout.eval()

            model.module.virus_gru.train()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics, train_loss = run_net_with_frozen_antibody_and_embedding(model, conf, loader_train, loss_fn, optimizer, isTrain=True)
            metrics_train_per_epochs.append(train_metrics)
            train_losses.append(train_loss)
            train_mcc.append(train_metrics[MATTHEWS_CORRELATION_COEFFICIENT])

            # model.module.virus_gru.eval()
            # model.module.fc_dropout.eval()
            # model.module.fully_connected.eval()

            # test_metrics, val_loss = eval_network_cv(model, loader_val, loss_fn)
            test_metrics, val_loss = eval_network_cv(model, conf, loader_val, loss_fn)
            metrics_test_per_epochs.append(test_metrics)
            val_mcc.append(test_metrics[MATTHEWS_CORRELATION_COEFFICIENT])
            val_losses.append(val_loss)

            # print(f"Epoch {epoch + 1} - Train Loss: {train_loss}, Validation Loss: {val_loss}")
            
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cross_validation_round + 1}.tar'))
            if log_every_epoch:
                logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        

        return metrics_train_per_epochs, metrics_test_per_epochs, best, train_mcc, val_mcc, train_losses, val_losses
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)


def train_with_frozen_antibody_and_embedding_GRU(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', save_model = True, log_every_epoch = True):
    # Freezing the embeddings and antibody subnetworks
    for param in model.module.aminoacid_embedding.parameters():
        param.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p = 0)
    model.module.embedding_dropout.requires_grad = False

    for param in model.module.light_ab_fc.parameters():
        param.requires_grad = False
    model.module.light_ab_dropout.dropout = 0
    model.module.light_ab_dropout.requires_grad = False

    for param in model.module.light_ab_att.parameters():
        param.requires_grad = False
    model.module.light_ab_att_dropout.dropout = 0
    model.module.light_ab_att_dropout.requires_grad = False

    for param in model.module.heavy_ab_fc.parameters():
        param.requires_grad = False
    model.module.heavy_ab_dropout.dropout = 0
    model.module.heavy_ab_dropout.requires_grad = False

    for param in model.module.heavy_ab_att.parameters():
        param.requires_grad = False
    model.module.heavy_ab_att_dropout.dropout = 0
    model.module.heavy_ab_att_dropout.requires_grad = False

    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = np.zeros(3)
    try:
        for epoch in range(epochs):
            model.module.aminoacid_embedding.eval()
            model.module.embedding_dropout.eval()
            model.module.light_ab_fc.eval()
            model.module.light_ab_dropout.eval()
            model.module.light_ab_att.eval()
            model.module.light_ab_att_dropout.eval()
            model.module.heavy_ab_fc.eval()
            model.module.heavy_ab_dropout.eval()
            model.module.heavy_ab_att.eval()
            model.module.heavy_ab_att_dropout.eval()

            model.module.virus_gru.train()
            model.module.fc_dropout.train()
            model.module.fully_connected.train()

            train_metrics = run_net_with_frozen_antibody_and_embedding_GRU(model, conf, loader_train, loss_fn, optimizer, isTrain = True)
            metrics_train_per_epochs.append(train_metrics)

            test_metrics = eval_network(model, loader_val)
            metrics_test_per_epochs.append(test_metrics)
            # We save a model chekpoint if we find any improvement
            if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
                best = test_metrics
                if save_model:
                    if cross_validation_round is None:
                        cross_validation_round=0
                    t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}_GRU_cv{cross_validation_round + 1}.tar'))
            if log_every_epoch:
                logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')

        logging.info(f'Cross validation round {cross_validation_round + 1}, Correlation: {best[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {best[ACCURACY]}')
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)



def train_network_n_times(model, conf, loader_train, loader_val, cross_validation_round, epochs, model_title = 'model', model_path = '', pruner: CrossValidationPruner = None):
    loss_fn = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = conf['LEARNING_RATE'])
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    milestones = np.floor(epochs * np.array([.25, .5, .75]))
    step_counter = 0
    try:
        for epoch in range(epochs):
            model.train()
            train_metrics = run_network_for_training(model, conf, loader_train, loss_fn, optimizer, epochs, pruner)
            metrics_train_per_epochs.append(train_metrics)
            if loader_val:
                test_metrics = eval_network(model, loader_val)
                metrics_test_per_epochs.append(test_metrics)
                logging.info(f'Epoch {epoch + 1}, Correlation: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {test_metrics[ACCURACY]}')
            else:
                metrics_train_per_epochs.append(train_metrics)
                logging.info(f'Epoch {epoch + 1}, Correlation: {train_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {train_metrics[ACCURACY]}')
            if epoch in milestones and pruner is not None:
                cv_fold = cross_validation_round if cross_validation_round is not None else 0
                # This throws a pruning exception if the trial needs to be pruned
                pruner.report(test_metrics[MATTHEWS_CORRELATION_COEFFICIENT], step_counter, cv_fold)
                step_counter += 1
        t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title}.tar'))
        last = test_metrics if loader_val else train_metrics
        cv_info = f'CV {cross_validation_round + 1}, ' if cross_validation_round is not None else ''
        logging.info(f'{cv_info}Correlation: {last[MATTHEWS_CORRELATION_COEFFICIENT]}, Accuracy: {last[ACCURACY]}')
        return metrics_train_per_epochs, metrics_test_per_epochs, last
    except KeyboardInterrupt as e:
        logging.info('Training interrupted at epoch ' + epoch)


def run_net_with_frozen_embeddings(model, conf, loader, loss_fn, optimizer=None, isTrain=False):
    """
    Freeze amino-acid embeddings; train antibody + virus modules.
    """
    metrics = np.zeros(3)
    total_weight, total_loss = 0.0, 0.0

    for ab_light, ab_heavy, virus, pngs_mask, y in loader:
        batch_size = len(y)

        # embeddings are constants
        with t.no_grad():
            ab_light_e, ab_heavy_e, virus_e = model.module.forward_embeddings(ab_light, ab_heavy, virus, batch_size)

        # antibody + virus are trainable
        ab_hidden = model.module.forward_antibodyes(ab_light_e, ab_heavy_e)
        pred = model.module.forward_virus(virus_e, pngs_mask, ab_hidden)

        if pred.shape != y.shape:
            pred = pred.reshape(y.shape)

        loss = loss_fn(pred, y)
        total_loss += loss.item() * batch_size

        if isTrain:
            assert optimizer is not None
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf['GRAD_NORM_CLIP'], norm_type=1)
            optimizer.step()
            optimizer.zero_grad()

        w = len(y) / conf['BATCH_SIZE']
        total_weight += w
        metrics += compute_metrics(to_numpy(y), to_numpy(pred)) * w

    avg_loss = total_loss / total_weight if total_weight else 0.0
    return metrics / total_weight, avg_loss


def train_with_frozen_embeddings_only(model, conf, loader_train, loader_val, cv_round, epochs,
                                      model_title='model', model_path='', save_model=True, log_every_epoch=True):
    """
    Freeze embeddings; unfreeze antibody (and keep virus trainable).

    Why this works:
        - Embeddings are computed under no_grad() and their parameters are requires_grad=False, so they’re frozen.
        - Antibody submodules run with grad; their parameters are included in the optimizer (we filter on requires_grad=True).
        - We restore antibody dropout to your configured value so training behavior matches your base config rather than the fully-frozen path that set dropout to 0.
        - Virus GRU / FC remain trainable as before.

    """
    # 1) Freeze embeddings
    for p in model.module.aminoacid_embedding.parameters():
        p.requires_grad = False
    model.module.embedding_dropout = t.nn.Dropout(p=0)  # keep disabled; no params anyway

    # 2) UNFREEZE antibody submodules (light/heavy FC + attention) and restore dropouts
    for p in model.module.light_ab_fc.parameters():   p.requires_grad = True
    for p in model.module.light_ab_att.parameters():  p.requires_grad = True
    for p in model.module.heavy_ab_fc.parameters():   p.requires_grad = True
    for p in model.module.heavy_ab_att.parameters():  p.requires_grad = True

    # restore antibody dropouts (they were set to 0 in the fully frozen path)
    if hasattr(model.module, 'light_ab_dropout'):      model.module.light_ab_dropout.dropout = conf['ANTIBODIES_DROPOUT']
    if hasattr(model.module, 'light_ab_att_dropout'):  model.module.light_ab_att_dropout.dropout = conf['ANTIBODIES_DROPOUT']
    if hasattr(model.module, 'heavy_ab_dropout'):      model.module.heavy_ab_dropout.dropout = conf['ANTIBODIES_DROPOUT']
    if hasattr(model.module, 'heavy_ab_att_dropout'):  model.module.heavy_ab_att_dropout.dropout = conf['ANTIBODIES_DROPOUT']

    # virus pathway stays trainable
    loss_fn   = t.nn.BCELoss()
    optimizer = t.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['LEARNING_RATE'])

    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    train_mcc, val_mcc, train_losses, val_losses = [], [], [], []
    best = np.zeros(3)

    for epoch in range(epochs):
        # modes
        model.module.aminoacid_embedding.eval()
        model.module.embedding_dropout.eval()

        model.module.light_ab_fc.train()
        model.module.light_ab_dropout.train()          if hasattr(model.module, 'light_ab_dropout') else None
        model.module.light_ab_att.train()
        model.module.light_ab_att_dropout.train()      if hasattr(model.module, 'light_ab_att_dropout') else None

        model.module.heavy_ab_fc.train()
        model.module.heavy_ab_dropout.train()          if hasattr(model.module, 'heavy_ab_dropout') else None
        model.module.heavy_ab_att.train()
        model.module.heavy_ab_att_dropout.train()      if hasattr(model.module, 'heavy_ab_att_dropout') else None

        model.module.virus_gru.train()
        model.module.fc_dropout.train()
        model.module.fully_connected.train()

        train_metrics, train_loss = run_net_with_frozen_embeddings(model, conf, loader_train, loss_fn, optimizer, isTrain=True)
        metrics_train_per_epochs.append(train_metrics); train_losses.append(train_loss)
        train_mcc.append(train_metrics[MATTHEWS_CORRELATION_COEFFICIENT])

        # validation
        test_metrics, val_loss = eval_network_cv(model, conf, loader_val, loss_fn)
        metrics_test_per_epochs.append(test_metrics); val_losses.append(val_loss)
        val_mcc.append(test_metrics[MATTHEWS_CORRELATION_COEFFICIENT])

        if test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best[MATTHEWS_CORRELATION_COEFFICIENT]:
            best = test_metrics
            if save_model:
                t.save({'model': model.state_dict()}, os.path.join(model_path, f'{model_title} cv {cv_round + 1}.tar'))

        if log_every_epoch:
            logging.info(f'Epoch {epoch+1}, MCC: {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]:.4f}, ACC: {test_metrics[ACCURACY]:.4f}')

    logging.info(f'CV {cv_round + 1}, Best MCC: {best[MATTHEWS_CORRELATION_COEFFICIENT]:.4f}, Best ACC: {best[ACCURACY]:.4f}')
    return (metrics_train_per_epochs, metrics_test_per_epochs, best,
            train_mcc, val_mcc, train_losses, val_losses)
