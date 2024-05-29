import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn import metrics
from tqdm import tqdm
import json
from model import DKVMN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_size = 20 
n_q = 123 
k_dim = 50 
v_m_dim = 200 
v_dim = 200
f_dim = 50 

lr = 0.01
epoch = 300
batch_size = 64
maxgradnorm = 50.0

def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0

def train_epoch(e, model, optimizer, q_data, qa_data):
    loss_function = nn.BCELoss()
    N = int(math.floor(len(q_data) / batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0

    model.train()

    for idx in tqdm(range(N), "Epoch %s" % e):
        q_batch = q_data[idx * batch_size: (idx + 1) * batch_size, :]
        qa_batch = qa_data[idx * batch_size: (idx + 1) * batch_size, :]
        target_batch = qa_data[idx * batch_size: (idx + 1) * batch_size, :]

        target_batch = (target_batch - 1) // n_q
        # target = np.floor(target)

        input_q = torch.LongTensor(q_batch).to(device)
        input_qa = torch.LongTensor(qa_batch).to(device)
        target_batch = torch.Tensor(target_batch).to(device)

        pred_batch = model.forward(input_q, input_qa) # (bs, 200)
        pred_1d = pred_batch.view(-1, 1) # [batch_size * seq_len, 1]

        target_1d = target_batch.view(-1, 1) # [batch_size * seq_len, 1]

        # 找出target大于或等于0为true，小于0为false，填充值为-1
        mask = target_1d.ge(0) 
           
        f_pred = torch.masked_select(pred_1d, mask)
        f_target = torch.masked_select(target_1d, mask)
        loss = F.binary_cross_entropy_with_logits(f_pred, f_target)
        f_pred = torch.sigmoid(f_pred)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), maxgradnorm)
        optimizer.step()
        epoch_loss += loss.item()

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

    return epoch_loss / N, auc, accuracy

def train(train_data, test_data=None):

    q_data, qa_data = train_data
    model = DKVMN(m_size, n_q, k_dim, v_m_dim, v_dim, f_dim)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.9))
    model.to(device)

    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    all_train_loss = {}
    best_valid_auc = 0

    for idx in range(epoch):
        loss, auc, acc = train_epoch(idx+1, model, optimizer, q_data, qa_data)
        print('Training ---- loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (loss, auc, acc))

        if test_data is not None:
            loss, auc, acc = eval(model, test_data)
            all_valid_loss[idx + 1] = loss
            all_valid_auc[idx + 1] = auc
            all_valid_accuracy[idx + 1] = acc
            all_train_loss[idx + 1] = loss

            print('Evaluating ---- loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (loss, auc, acc))
            logging.info(f"[Epoch {idx + 1}] valid_auc: {auc}")
            if auc > best_valid_auc:
                print('valid auc improve: %3.4f to %3.4f' % (best_valid_auc, auc))
                best_valid_auc = auc
            
    with open("list_for_draw_dkvmn.json", "w") as f:
        json.dump(all_train_loss ,f)  
        f.write("\n")          
        json.dump(all_valid_loss, f)
        f.write("\n")
        json.dump(all_valid_accuracy, f)
        f.write("\n")
        json.dump(all_valid_auc ,f)

def eval(model, data):
    q_data, qa_data = data
    loss_function = nn.BCELoss()
    N = int(math.floor(len(q_data) / batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(N), "Epoch %s" % epoch):
            q_batch = q_data[idx * batch_size: (idx + 1) * batch_size, :]
            qa_batch = qa_data[idx * batch_size: (idx + 1) * batch_size, :]
            target_batch = qa_data[idx * batch_size: (idx + 1) * batch_size, :]

            target_batch = (target_batch - 1) // n_q
            # target = np.floor(target)

            input_q = torch.LongTensor(q_batch).to(device)
            input_qa = torch.LongTensor(qa_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)

            pred_batch = model.forward(input_q, input_qa) # (bs, 200)
            pred_1d = pred_batch.view(-1, 1) # [batch_size * seq_len, 1]

            target_to_1d = torch.chunk(target_batch, batch_size, 0)
            target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
            target_1d = target_1d.permute(1, 0) # [batch_size * seq_len, 1]

            # 找出target大于或等于0为true，小于0为false，填充值为-1
            mask = target_1d.ge(0) 
            
            f_pred = torch.masked_select(pred_1d, mask)
            f_target = torch.masked_select(target_1d, mask)
            loss = F.binary_cross_entropy_with_logits(f_pred, f_target)
            f_pred = torch.sigmoid(f_pred)

            epoch_loss += loss.item()

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

    return epoch_loss / N, auc, accuracy

def save(self, filepath):
    torch.save(self.model.state_dict(), filepath)
    logging.info("save parameters to %s" % filepath)

def load(self, filepath):
    self.model.load_state_dict(torch.load(filepath))
    logging.info("load parameters from %s" % filepath)