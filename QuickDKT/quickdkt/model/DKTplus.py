import torch
from torch import nn
from ..utils.meta import KTM
import logging
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


class DKTNet(nn.Module):
    def __init__(self, question_num, hidden_size, dropout=0.0) -> None:
        super(DKTNet, self).__init__()
        self.hidden_dim = hidden_size
        self.rnn = nn.RNN(
                input_size = question_num * 2, 
                hidden_size = hidden_size,
                num_layers = 1,
                batch_first = True,
                nonlinearity = 'tanh'
                )

        self.fc = nn.Linear(self.hidden_dim, question_num)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        """
        inputs:
            input: (batch_size, sequence_length, input_size)
            h_0: (D * num_layers, batch_size, hidden_size).Defaults to zeros if not provided
        outputs:
            output: (batch_size, sequence_length, D * hidden_size)
            h_n: (D * num_layers, batch_size, hidden_size)
            containing the final hidden state for each element in the batch
        """
        output, h_n = self.rnn(input)
        output = self.sig(self.fc(self.dropout(output)))
        return output, h_n
    
def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    # questions = torch.nonzero(raw_question_matrix)[0:, 1] % num_questions
    # 获取0-48题的pred值，分别作为1-49题的预测值使用
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    # 确定pred的真实长度，以去除填充值
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth

def get_loss_params(raw_pred, batch, num_questions, loss_params):
    lr = loss_params['lr']
    lw1 = loss_params['lw1']
    lw2 = loss_params['lw2']
    batch_size = batch.shape[0]
    wr, w1, w2 = 1e-12, 1e-12, 1e-12

    # Create an instance of BCELoss
    bce_loss = nn.BCELoss()

    for student in range(batch_size):
        raw_qusetion_matrix = batch[student]
        raw_pred_one = raw_pred[student]
        # 获取每个batch中0-49题one-hot放入model中的预测值，最大长度为50
        questions = torch.nonzero(raw_qusetion_matrix)[:, 1] % num_questions
        length = questions.shape[0]
        # 该batch只有一行数据有效，无法参与损失值计算
        if length == 1:
            continue
        pred_matrix = raw_pred_one[: length]
        pred = pred_matrix.gather(1, questions.view(-1, 1)).flatten()
        # 获取每个batch中0-49题one-hot还原成题目正误的label，最大长度50
        truth = torch.nonzero(raw_qusetion_matrix)[:, 1] // num_questions
        truth = truth.float()
        # BCELoss要求预测值和目标值均为浮点类型
        if lw1 > 0.0 or lw2 > 0.0:
            post_pred = pred_matrix[1:, :]
            pre_pred = pred_matrix[:-1, :]
            diff = post_pred - pre_pred
            # diff: (49, 123)
            question_n = diff.shape[1]
            temp = torch.norm(diff, 1, 1)/diff.shape[1]
            temp1 = torch.mean(temp)
            w1 += torch.mean(torch.norm(diff, 1, 1)) / diff.shape[1]
            w2 += torch.mean(torch.norm(diff, 2, 1) ** 2) / diff.shape[1]
            if torch.isnan(w1).any() or torch.isnan(w2).any():
                print("NaN detected i norm calculations")
        else:
            w1 = 0.0
            w2 = 0.0

        if lr > 0.0:
            wr += bce_loss(pred, truth)
        else:
            wr = 0.0
    wr = wr / batch_size
    w1 = w1 / batch_size
    w2 = w2 / batch_size   
    loss_regu_items =  lr * wr + lw1 * w1 + lw2 * w2
    return loss_regu_items

# 定义早停法
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience # 允许无改善的最大轮数
        self.counter = 0 # 当前无改善的轮次计数
        self.best_score = None # 最佳得分
        # self.best_epoch = None # 最佳训练代数
        self.early_stop = False # 是否早停

    def __call__(self, auc_score, model):
        if self.best_score is None:
            self.best_score = auc_score
            model.save("dktplus.params") # 保存当前模型和状态
        elif auc_score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = auc_score
            model.save("dktplus.params") # 保存当前模型和状态
            self.counter = 0   

def get_w(raw_pred, batch, num_questions):
    bce_loss = nn.BCELoss()
    batch_size = batch.shape[0]
    wr, w1, w2 = 1e-12, 1e-12, 1e-12

    for student in range(batch_size):
        raw_qusetion_matrix = batch[student]
        raw_pred_one = raw_pred[student]
        # 获取每个batch中0-49题one-hot放入model中的预测值，最大长度为50
        questions = torch.nonzero(raw_qusetion_matrix)[:, 1] % num_questions
        length = questions.shape[0]
        # 该batch只有一行数据有效，无法参与损失值计算
        if length == 1:
            continue
        pred_matrix = raw_pred_one[: length]
        pred = pred_matrix.gather(1, questions.view(-1, 1)).flatten()
        # 获取每个batch中0-49题one-hot还原成题目正误的label，最大长度50
        truth = torch.nonzero(raw_qusetion_matrix)[:, 1] // num_questions
        truth = truth.float()

        post_pred = pred_matrix[1:, :]
        pre_pred = pred_matrix[:-1, :]
        diff = post_pred - pre_pred
        # diff: (49, 123)
        question_n = diff.shape[1]
        temp = torch.norm(diff, 1, 1)/diff.shape[1]
        temp1 = torch.mean(temp)
        w1 += torch.mean(torch.norm(diff, 1, 1)) / diff.shape[1]
        w2 += torch.mean(torch.norm(diff, 2, 1) ** 2) / diff.shape[1]
        wr += bce_loss(pred, truth)
        if torch.isnan(w1).any() or torch.isnan(w2).any():
            print("NaN detected i norm calculations")

    wr = wr / batch_size
    w1 = w1 / batch_size
    w2 = w2 / batch_size   
    return wr, w1, w2

def process_for_AUC_N(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    # questions = torch.nonzero(raw_question_matrix)[0:, 1] % num_questions
    # 获取当前的题号
    questions = torch.nonzero(raw_question_matrix)[:, 1] % num_questions
    # 确定pred的真实长度，以去除填充值
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[:, 1] // num_questions
    return pred, truth

class DKTPlus(KTM):
    def __init__(self, question_num, hidden_size, loss_params=None):
        super(DKTPlus, self).__init__()
        self.question_num = question_num
        self.dkt_net = DKTNet(question_num, hidden_size)
        self.dkt_net = self.dkt_net.cuda()
        self.loss_params = loss_params if loss_params is not None else {}

    def train(self, train_data, test_data=None, *, epoch, device="cuda", lr=1e-3) -> torch.Any:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_net.parameters(), lr)
        
        # 定义早停
        early_stopping = EarlyStopping(patience=5)
        epoch_auc = []
        for e in range(epoch):
            # sets the model to train mode at the beginning of each epoch
            self.dkt_net.train()
            # 计算每个batch的损失记录
            epoch_loss = []
            for batch in tqdm.tqdm(train_data, "Epoch %s" % (e+1)):
                batch: torch.Tensor = batch.to(device)
                integrated_pred, _ = self.dkt_net(batch)
                integrated_pred: torch.Tensor = integrated_pred.to(device)
                batch_size = batch.shape[0]
                batch_pred, batch_target = torch.Tensor([]), torch.Tensor([])
                batch_pred = batch_pred.to(device)
                batch_target = batch_target.to(device)
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.question_num)
                    batch_pred = torch.cat([batch_pred, pred])
                    batch_target = torch.cat([batch_target, truth.float()])
                
                # 正则化项
                regular_items = get_loss_params(integrated_pred, batch, self.question_num, self.loss_params)
                loss = loss_function(batch_pred, batch_target) + regular_items
                epoch_loss.append(loss.item())
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("[Epoch %d] LogisticLoss: %.6f" % (e+1, np.mean(epoch_loss)))


            # 验证集
            if test_data is not None:
                auc, acc = self.eval(test_data)
                logging.info(f"Epoch{e+1} auc: {auc}, acc: {acc}")
                # 判断早停条件
                early_stopping(auc, self)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    print("best_auc: %.6f" % early_stopping.best_score)
                    wr, w1, w2 = self.compute_w(test_data)
                    auc_current = self.eval_current(test_data)
                    logging.info(f"Epoch{e+1} auc_current: {auc_current}")
                    # logging.info(f"wr: {wr}, w1: {w1}, w2: {w2}" )
                    # 到达早停条件计算lambda_w1和lambda_w2
                    break
                epoch_auc.append(auc)
                print("[Epoch %d] auc: %.6f" % (e+1, auc))      
        return epoch_auc, np.mean(epoch_loss) # 返回最后一代的损失

    def eval(self, test_data, device="cuda") -> float:
        self.dkt_net.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        y_pred = y_pred.to(device)
        y_truth = y_pred.to(device)
        for batch in tqdm.tqdm(test_data, "evaluating"):
            batch: torch.Tensor = batch.to(device)
            integrated_pred, _ = self.dkt_net(batch)
            integrated_pred: torch.Tensor = integrated_pred.to(device)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.question_num)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
        # can't convert cuda:0 device type tensor to numpy
        y_pred = y_pred.detach().cpu().numpy()
        y_truth = y_truth.detach().cpu().numpy()

        valid_auc = roc_auc_score(y_truth, y_pred) 
        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0
        valid_acc = accuracy_score(y_truth, y_pred)
        return valid_auc, valid_acc
    # 计算AUC(C)

    def eval_current(self, test_data, device="cuda") -> float:
        self.dkt_net.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        y_pred = y_pred.to(device)
        y_truth = y_pred.to(device)
        for batch in tqdm.tqdm(test_data, "evaluating"):
            batch: torch.Tensor = batch.to(device)
            integrated_pred, _ = self.dkt_net(batch)
            integrated_pred: torch.Tensor = integrated_pred.to(device)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_for_AUC_N(batch[student], integrated_pred[student], self.question_num)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
        # can't convert cuda:0 device type tensor to numpy
        y_pred = y_pred.cpu()
        y_truth = y_truth.cpu()
        return roc_auc_score(y_truth.detach().numpy(), y_pred.detach().numpy())
    
    def save(self, filepath):
        torch.save(self.dkt_net.state_dict(), filepath)
        # logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


    def compute_w(self, test_data, device="cuda"):
        self.dkt_net.eval()
        y_pred = torch.Tensor([])
        y_truth = torch.Tensor([])
        y_pred = y_pred.to(device)
        y_truth = y_pred.to(device)
        wr_list = []
        w1_list = []
        w2_list = []
        for batch in tqdm.tqdm(test_data, "evaluating_w"):
            batch: torch.Tensor = batch.to(device)
            integrated_pred, _ = self.dkt_net(batch)
            integrated_pred: torch.Tensor = integrated_pred.to(device)
            batch_size = batch.shape[0]
            wr, w1 ,w2 = get_w(integrated_pred, batch, self.question_num)
            wr_list.append(wr.item())
            w1_list.append(w1.item())
            w2_list.append(w2.item())
        wr = np.mean(wr_list)
        w1 = np.mean(w1_list)
        w2 = np.mean(w2_list)

        return wr, w1, w2