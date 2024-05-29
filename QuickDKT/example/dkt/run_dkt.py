# coding: utf-8
# 2024/5/21 @ shucunwang

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import argparse
import logging
sys.path.insert(0, "../..")

from load_data import PID_DATA
from quickdkt.train.train_dkt import dkt
from quickdkt.model import DKT
from quickdkt.data import AKTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(args):

    dat = PID_DATA(n_question=args.n_q, seqlen=args.seq_len, separate_char=',') 
    # train_data_path = "../../data/2009_skill_builder_data_corrected/assistchall_train.txt"
    # test_data_path = "../../data/2009_skill_builder_data_corrected/assistchall_test.txt"
    train_data_path = "../../data/anonymized_full_release_competition_dataset/assistchall_train_pid.txt"
    test_data_path = "../../data/anonymized_full_release_competition_dataset/assistchall_test_pid.txt"

    # dataformat: tuple (q_data_array, qa_data_array)
    train_data = dat.load_data(train_data_path)
    test_data = dat.load_data(test_data_path)

    train = AKTDataset(train_data, args.n_q)
    test = AKTDataset(test_data, args.n_q)
    train_loader = DataLoader(train, args.bs, shuffle=True)
    test_loader = DataLoader(test, args.bs, shuffle=False)

    # log config
    logging.basicConfig(filename='training_dkt_promblem.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # model config
    model = DKT(args.n_q, args.n_s, args.input_size, args.hidden_size, 
                args.cell_type, device=args.device)
    model.to(device)
    loss_fc = F.binary_cross_entropy_with_logits
    optim = torch.optim.Adam(params = model.parameters(), 
                             lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8)
    ktm = dkt(model, device)

    # start training
    best_valid_auc = 0.0
    for idx in range(args.epoch):
        loss, auc, acc = ktm.train_epoch(idx+1, optim, loss_fc, train_loader)
        print('Training --- loss : %3.5f, auc : %3.5f, accuracy : %3.5f' 
        % (loss, auc, acc))
        logging.info(f"Training--epoch[{str(idx+1)}]--loss: {loss}, auc: {auc}, acc: {acc}")

        loss, auc, acc = ktm.eval_epoch(idx+1, loss_fc, test_loader)
        print('Evaluating --- loss : %3.5f, auc : %3.5f, accuracy : %3.5f' 
        % (loss, auc, acc))
        logging.info(f"Evaluating--epoch[{str(idx+1)}]--loss: {loss}, auc: {auc}, acc: {acc}")

        if auc > best_valid_auc:
            print('valid auc improve: %3.4f to %3.4f' % (best_valid_auc, auc))
            best_valid_auc = auc

        # scheduler.step()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train dkvmn model")

    # model args
    parser.add_argument("--cell_type", type=str, default="lstm")
    parser.add_argument("--n_q", type=int, default=3162)
    parser.add_argument("--n_s", type=int, default=102)
    parser.add_argument("--input_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=200)

    # train args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=35)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--device", type=str, default=device)

    args = parser.parse_args()
    run(args)