# coding: utf-8
# 2024/5/13 @ shucunwang

import logging
import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics
from tqdm import tqdm
from ..utils.meta import KTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maxgradnorm = 50.0

class dkvmn(KTM):
    def __init__(self, model, n_q):
        super().__init__()
        self.model = model
        self.n_q = n_q

    def train_epoch(self, epoch, optim, loss_fc, train_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.train()

        for (q, qa, target) in tqdm(train_loader, "Epoch %s" % epoch):
            q, qa, target = (
                q.to(device),
                qa.to(device),
                target.to(device),
            )
            mask = target >= 0
            mask.to(device)

            pred = self.model(q, qa) # (bs, 200)
            f_pred = torch.masked_select(pred, mask)
            f_target = torch.masked_select(target, mask)
            loss = loss_fc(f_pred, f_target)
            f_pred = torch.sigmoid(f_pred)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), maxgradnorm)
            optim.step()
            epoch_loss.append(loss.item())

            pred = f_pred.detach().cpu().numpy()
            target = f_target.detach().cpu().numpy()
            
            pred_list.append(pred)
            target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = metrics.roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy

    def eval_epoch(self, epoch, loss_fc, test_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.eval()

        with torch.no_grad():
            for (q, qa, target) in tqdm(test_loader, "Epoch %s" % epoch):
                q, qa, target = (
                    q.to(device),
                    qa.to(device),
                    target.to(device),
                )
                mask = target >= 0
                mask.to(device)

                pred = self.model(q, qa) # (bs, 200)
                f_pred = torch.masked_select(pred, mask)
                f_target = torch.masked_select(target, mask)
                loss = loss_fc(f_pred, f_target)
                f_pred = torch.sigmoid(f_pred)
                epoch_loss.append(loss.item())

                pred = f_pred.detach().cpu().numpy()
                target = f_target.detach().cpu().numpy()
                
                pred_list.append(pred)
                target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = metrics.roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)