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

class dkt(KTM):
    def __init__(self, model, device):
        super(dkt, self).__init__()
        self.model = model
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
            q_one_hot = F.one_hot(q, num_classes=103)[:, 1:, :]
            q_mask = q_one_hot > 0
            temp_pred = pred[:, :-1, :]

            temp_pred = torch.masked_select(temp_pred, q_mask).view(bs, -1)
            f_target = torch.masked_select(target[:, 1:], mask[:, 1:])
            f_pred = torch.masked_select(temp_pred, mask[:, 1:])

            loss = loss_fc(f_pred, f_target)
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

    # def train_epoch(self, idx, train_data, optimizer, loss_function, device):
    #     epoch_loss = []
    #     all_pred = torch.LongTensor([]).to(device)
    #     all_target = torch.LongTensor([]).to(device)
    #     for batch in tqdm.tqdm(train_data, "Epoch %s" % (idx)):
    #         batch: torch.Tensor = batch.to(device)
    #         integrated_pred, _ = self.dkt_net(batch)
    #         integrated_pred: torch.Tensor = integrated_pred.to(device)
    #         batch_size = batch.shape[0]
    #         batch_pred, batch_target = torch.Tensor([]), torch.Tensor([])
    #         batch_pred = batch_pred.to(device)
    #         batch_target = batch_target.to(device)
    #         for student in range(batch_size):
    #             pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
    #             batch_pred = torch.cat([batch_pred, pred])
    #             batch_target = torch.cat([batch_target, truth.float()])
            
    #         all_pred = torch.cat([all_pred, batch_pred])
    #         all_target = torch.cat([all_target, batch_target])                
    #         loss = loss_function(batch_pred, batch_target)

    #         epoch_loss.append(loss.item())
    #         # backward
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     train_loss = np.mean(epoch_loss)
    #     return train_loss, all_pred, all_target

    # def train(self, epoch, train_data, valid_data=None, lr=1e-3) -> ...:
    #     device = self.device
    #     loss_function = nn.BCELoss()
    #     optimizer = torch.optim.Adam(self.dkt_net.parameters(), lr)

    #     # 画图list
    #     all_train_loss = []
    #     all_valid_loss = []
    #     epoch_auc = []
    #     epoch_acc = []
    #     for e in range(epoch):
    #         self.dkt_net.train()
    #         # 计算每个batch的损失记录
    #         loss , all_pred, all_target = self.train_one_epoch(e+1, train_data, optimizer, loss_function, device)
    #         # 训练集评估
    #         all_pred = np.asarray(all_pred.tolist())
    #         all_target = np.asarray(all_target.tolist())
    #         train_auc = roc_auc_score(all_target, all_pred) 
    #         all_pred[all_pred >= 0.5] = 1.0
    #         all_pred[all_pred < 0.5] = 0.0
    #         train_acc = accuracy_score(all_target, all_pred) 
            
    #         # 验证集评估
    #         if valid_data is not None:

    #             valid_loss, valid_auc, valid_acc = self.eval(valid_data)
    #             print("[Epoch %d] valid_loss: %3.5f, valid_auc: %3.5f, valid_acc : %3.5f" % 
    #               (e+1, valid_loss, valid_auc, valid_acc))
    #             logging.info(f"Evaluating--epoch[{str(e+1)}]--loss: {valid_loss}, auc: {valid_auc}, acc: {valid_acc}")
    #             all_valid_loss.append(valid_loss)

    # def eval(self, test_data) -> float:
    #     self.dkt_net.eval()
    #     device = self.device
    #     loss_function = nn.BCELoss()
    #     y_pred = torch.Tensor([])
    #     y_truth = torch.Tensor([])
    #     y_pred = y_pred.to(device)
    #     y_truth = y_pred.to(device)
    #     for batch in tqdm.tqdm(test_data, "evaluating"):
    #         batch: torch.Tensor = batch.to(device)
    #         integrated_pred, _ = self.dkt_net(batch)
    #         integrated_pred: torch.Tensor = integrated_pred.to(device)
    #         batch_size = batch.shape[0]
    #         for student in range(batch_size):
    #             pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
    #             y_pred = torch.cat([y_pred, pred])
    #             y_truth = torch.cat([y_truth, truth])
            
    #     valid_loss = loss_function(y_pred, y_truth).item()
    #     y_pred = np.asarray(y_pred.tolist())
    #     y_truth = np.asarray(y_truth.tolist())
    #     valid_auc = roc_auc_score(y_truth, y_pred) 
    #     y_pred[y_pred >= 0.5] = 1.0
    #     y_pred[y_pred < 0.5] = 0.0
    #     valid_acc = accuracy_score(y_truth, y_pred) 

    #     return valid_loss, valid_auc, valid_acc
    
    # def save(self, filepath):
    #     torch.save(self.dkt_net.state_dict(), filepath)
    #     # logging.info("save parameters to %s" % filepath)

    # def load(self, filepath):
    #     self.dkt_net.load_state_dict(torch.load(filepath))
    #     # logging.info("load parameters from %s" % filepath)