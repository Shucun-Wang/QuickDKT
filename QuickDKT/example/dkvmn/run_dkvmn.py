# coding: utf-8
# 2024/5/13 @ shucun wang

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import argparse
import logging
sys.path.insert(0, "../..")

from load_data import Data
from quickdkt.train.train_dkvmn import dkvmn
from quickdkt.model import DKVMN
from quickdkt.data import KTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def run(args):

    dat = Data(n_question=args.n_q, seqlen=args.seq_len, separate_char=',') 
    # train_data_path = "../../data/2009_skill_builder_data_corrected/assistchall_train.txt"
    # test_data_path = "../../data/2009_skill_builder_data_corrected/assistchall_test.txt"
    train_data_path = "../../data/anonymized_full_release_competition_dataset/assistchall_train.txt"
    test_data_path = "../../data/anonymized_full_release_competition_dataset/assistchall_test.txt"

    # dataformat: tuple (q_data_array, qa_data_array)
    train_data = dat.load_data(train_data_path)
    test_data = dat.load_data(test_data_path)

    train = KTDataset(train_data, args.n_q)
    test = KTDataset(test_data, args.n_q)
    train_loader = DataLoader(train, args.bs, shuffle=True)
    test_loader = DataLoader(test, args.bs, shuffle=False)

    # log config
    logging.basicConfig(filename='training_dkvmn_promblem.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # model config
    model = DKVMN(args.m_size, args.n_q, args.k_dim, 
                  args.v_m_dim, args.v_dim, args.f_dim)
    model.to(device)
    loss_fc = F.binary_cross_entropy_with_logits
    optim = torch.optim.Adam(params = model.parameters(), 
                             lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8)
    ktm = dkvmn(model, args.n_q)

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
    parser.add_argument("--m_size", type=int, default=20)
    parser.add_argument("--n_q", type=int, default=123 )
    parser.add_argument("--k_dim", type=int, default=50)
    parser.add_argument("--v_m_dim", type=int, default=200)
    parser.add_argument("--v_dim", type=int, default=200)
    parser.add_argument("--f_dim", type=int, default=50)

    # train args
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=200)

    args = parser.parse_args()
    run(args)