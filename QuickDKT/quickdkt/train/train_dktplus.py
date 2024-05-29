# coding: utf-8
# 2024/5/21 @ shucunwang

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import tqdm
import logging

from ..utils.meta import KTM

class dktplus(KTM):
    def __init__(self, model, loss_params, device):
        super(dktplus, self).__init__()
        self.model = model
        self.loss_params = loss_params
        self.device = device

    def train_epoch(self, epoch, optim, loss_fc, train_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.train()

        for (q, qa, p, target) in tqdm.tqdm(train_loader, "Epoch %s" % epoch):

            bs = q.shape[0]
            q, qa, p, target = (
                q.to(self.device),
                qa.to(self.device),
                p.to(self.device),
                target.to(self.device),
            )
            mask = target >= 0
            mask.to(self.device)

            pred, _ = self.model(qa)
            q_one_hot = F.one_hot(q, num_classes=103)
            q_mask = q_one_hot > 0
            regluarization = get_regluarization(pred, target, q_mask, mask, self.loss_params, loss_fc)

            q_mask = q_mask[:, 1:, :]
            temp_pred = pred[:, :-1, :]
            temp_pred = torch.masked_select(temp_pred, q_mask).view(bs, -1)
            f_target = torch.masked_select(target[:, 1:], mask[:, 1:])
            f_pred = torch.masked_select(temp_pred, mask[:, 1:])

            loss = loss_fc(f_pred, f_target) + regluarization
            f_pred = torch.sigmoid(f_pred)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())

            pred = f_pred.detach().cpu().numpy()
            target = f_target.detach().cpu().numpy()
            
            pred_list.append(pred)
            target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy

    def eval_epoch(self, epoch, loss_fc, test_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.eval()

        with torch.no_grad():
            for (q, qa, p, target) in tqdm.tqdm(test_loader, "Epoch %s" % epoch):
                bs = q.shape[0]
                q, qa, p, target = (
                    q.to(self.device),
                    qa.to(self.device),
                    p.to(self.device),
                    target.to(self.device),
                )
                mask = target >= 0
                mask.to(self.device)

                pred, _ = self.model(qa)
                q_one_hot = F.one_hot(q, num_classes=103)[:, 1:, :]
                q_mask = q_one_hot > 0
                temp_pred = pred[:, :-1, :]

                temp_pred = torch.masked_select(temp_pred, q_mask).view(bs, -1)
                f_target = torch.masked_select(target[:, 1:], mask[:, 1:])
                f_pred = torch.masked_select(temp_pred, mask[:, 1:])
                
                f_pred = torch.sigmoid(f_pred)
                loss = loss_fc(f_pred, f_target)
                epoch_loss.append(loss.item())

                pred = f_pred.detach().cpu().numpy()
                target = f_target.detach().cpu().numpy()
                
                pred_list.append(pred)
                target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

def get_regluarization(pred, target, q_mask, mask, loss_params, loss_fc):
    bs = pred.shape[0]
    lambda_r  = loss_params["lambda_r"]
    lambda_w1 = loss_params["lambda_w1"]
    lambda_w2 = loss_params["lambda_w2"]

    temp_pred = torch.masked_select(pred, q_mask).view(bs, -1)
    f_target = torch.masked_select(target, mask)
    f_pred = torch.masked_select(temp_pred, mask)
    r = loss_fc(f_pred, f_target) / bs 

    pred = torch.sigmoid(pred)
    pre_pred = pred[:, :, :-1]
    post_pred = pred[:, :, 1: ]
    diff = post_pred - pre_pred 
    w1 = torch.norm(diff, dim=2, p=1).mean() 
    w2 = torch.norm(diff, dim=2, p=2).mean()

    regluarization = lambda_r * r + lambda_w1 * w1 + lambda_w2 * w2
    return regluarization