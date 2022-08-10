#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Losses. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import Registry

import utils.misc as misc
import utils.distributed as du

from models.utils.localization_losses import LOCALIZATION_LOSSES
from datasets.utils.mixup import label_smoothing
import models.utils.contrastive_losses as cont

SSL_LOSSES = Registry("SSL_Losses")

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "soft_target": SoftTargetCrossEntropy,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def calculate_loss(cfg, preds, logits, labels, cur_epoch):
    """
    Calculates loss according to cfg.
    For pre-training, losses are defined and registered in `SSL_LOSSES`.
    Different losses can be combined by specifying in the `cfg.PRETRAIN.LOSS` by
    connecting different loss names with `+`.
    
    For supervised training, this function supports cross entropy with mixup,
    label smoothing and the plain cross entropy.
    """
    loss_in_parts = {}
    weight = None
    if cfg.PRETRAIN.ENABLE:
        loss = 0
        loss_parts = cfg.PRETRAIN.LOSS.split('+')
        loss_weights = cfg.PRETRAIN.LOSS_WEIGHTS
        # sum up all loss items
        for loss_idx, loss_item in enumerate(loss_parts):
            loss_cur, weight = SSL_LOSSES.get("Loss_"+loss_item)(cfg, preds, logits, labels["self-supervised"], cur_epoch)
            if isinstance(loss_cur, dict):
                for k, v in loss_cur.items():
                    loss_in_parts[k] = v
                    if "debug" not in k and isinstance(v, torch.Tensor):
                        loss += loss_weights[loss_idx]*loss_in_parts[k]
            else:
                loss_in_parts[loss_item] = loss_cur
                loss += loss_weights[loss_idx]*loss_in_parts[loss_item]
    elif cfg.LOCALIZATION.ENABLE:
        loss = 0
        loss_parts = cfg.LOCALIZATION.LOSS.split('+')
        loss_weights = cfg.LOCALIZATION.LOSS_WEIGHTS
        for loss_idx, loss_item in enumerate(loss_parts):
            loss_cur, weight = LOCALIZATION_LOSSES.get("Loss_"+loss_item)(cfg, preds, logits, labels, cur_epoch)
            if isinstance(loss_cur, dict):
                for k, v in loss_cur.items():
                    loss_in_parts[k] = v
                    if "debug" not in k:
                        loss += loss_weights[loss_idx]*loss_in_parts[k]
            else:
                loss_in_parts[loss_item] = loss_cur
                loss += loss_weights[loss_idx]*loss_in_parts[loss_item]
    else:
        # Explicitly declare reduction to mean.
        loss_fun = get_loss_func(cfg.TRAIN.LOSS_FUNC)(reduction="mean")
        
        # Compute the loss.
        if "supervised_mixup" in labels.keys():
            if isinstance(labels["supervised_mixup"], dict):
                loss = 0
                for k, v in labels["supervised_mixup"].items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels["supervised_mixup"])
        else:
            if cfg.AUGMENTATION.LABEL_SMOOTHING > 0.0:
                labels_ = label_smoothing(cfg, labels["supervised"])
            else:
                labels_ = labels["supervised"]
            if isinstance(labels_, dict):
                loss = 0
                for k, v in labels_.items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels_)

    return loss, loss_in_parts, weight

@SSL_LOSSES.register()
def Loss_MoSIX(cfg, preds, logits, labels, cur_epoch=0): # Camera Movement Spatial Transform
    """
    Computes only-x MoSI loss.
    See Ziyuan Huang et al.
    Self-supervised Motion Learning from Static Images.
    https://arxiv.org/pdf/2104.00240

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions for the movement on the x-axis.
        logits (Tensor): the defined so that the function has the same form
            as other losses.
        labels (Dict): labels for the movement on the x-axis.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    b, c = preds["move_x"].shape
    pred_move_x = preds["move_x"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_move_x"] = loss_func(pred_move_x, labels["move_joint"].reshape(pred_move_x.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_MoSIY(cfg, preds, logits, labels, cur_epoch=0):
    """
    Computes only-y MoSI loss.
    See Ziyuan Huang et al.
    Self-supervised Motion Learning from Static Images.
    https://arxiv.org/pdf/2104.00240

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions for the movement on the y-axis.
        logits (Tensor): the defined so that the function has the same form
            as other losses.
        labels (Dict): labels for the movement on the y-axis.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    b, c = preds["move_y"].shape
    pred_move_y = preds["move_y"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_move_y"] = loss_func(pred_move_y, labels["move_joint"].reshape(pred_move_y.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_MoSIJoint(cfg, preds, logits, labels, cur_epoch=0):
    """
    Computes joint MoSI loss.
    See Ziyuan Huang et al.
    Self-supervised Motion Learning from Static Images.
    https://arxiv.org/pdf/2104.00240

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions for the joint movement.
        logits (Tensor): the defined so that the function has the same form
            as other losses.
        labels (Dict): labels for the joint movement.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    b, c = preds["move_joint"].shape
    pred_move_joint = preds["move_joint"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_joint"] = loss_func(pred_move_joint, labels["move_joint"].reshape(pred_move_joint.shape[0]))
    return loss, None


@SSL_LOSSES.register()
def Loss_Contrastive(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    batch_size_per_gpu, samples = labels["contrastive"].shape
    if isinstance(logits, list):
        logits = logits[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits = du.all_gather([logits])[0]
    else:
        all_logits = logits
    batch_size = all_logits.shape[0]//samples
    logits = construct_logits_with_gradient(logits, all_logits, batch_size_per_gpu, samples)
    loss["loss_contrastive"], loss["pos"], loss["neg"] = cont.contrastive_instance_discrimination(
        cfg, logits, batch_size, samples
    )
    loss["loss_single"] = loss["loss_contrastive"].item()
    loss["loss_contrastive"] = loss["loss_contrastive"] * du.get_world_size()
    return loss, None


@SSL_LOSSES.register()
def Loss_HiCo(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    batch_size_per_gpu, samples = labels["contrastive"].shape
    if isinstance(logits, list):
        logits = logits[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits = du.all_gather([logits])[0]
    else:
        all_logits = logits
    batch_size = all_logits.shape[0]//samples
    logits = construct_logits_with_gradient(logits, all_logits, batch_size_per_gpu, samples)
    loss["total_loss"], loss["pos"], loss["neg"], loss["vcl_loss"], loss["tcl_loss"] = cont.contrastive_hico(
        cfg, preds, logits, batch_size, samples
    )
    loss["total_loss"] = loss["total_loss"]
    return loss, None


def construct_logits_with_gradient(cur_logits, all_logits, batch_size_per_gpu, samples):
    num_nodes = du.get_world_size()
    rank = du.get_rank()
    num_samples_per_gpu = batch_size_per_gpu * samples
    if rank == 0:
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((cur_logits, logits_post), dim=0)
    elif rank == num_nodes-1:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        return torch.cat((logits_prev, cur_logits), dim=0)
    else:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((logits_prev, cur_logits, logits_post), dim=0)
