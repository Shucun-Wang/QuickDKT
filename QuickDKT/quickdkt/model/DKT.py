# coding: utf-8
# 2024/5/21 @ shucunwang

import torch
from torch import nn

class DKT(nn.Module):
    def __init__(self, n_q, n_s, input_size, hidden_size, cell_type, 
                 num_layers=1, dropout=0.0, device="cpu"):
        super(DKT, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = n_s + 1
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.device = device
        
        self.qa_embed = nn.Embedding(2 * n_q + 1, self.input_size)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.input_size, 
                self.hidden_size, 
                num_layers, 
                batch_first=True, 
                dropout=dropout
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.input_size, 
                self.hidden_size, 
                num_layers, 
                batch_first=True, 
                dropout=dropout                
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.input_size, 
                self.hidden_size, 
                num_layers, 
                batch_first=True, 
                dropout=dropout                  
            )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sig = nn.Sigmoid()

        if self.rnn is None:
            raise ValueError("Cell type only support LSTM, GRU or RNN type!")
        
    def forward(self, qa, state_in=None):

        batch_size = qa.shape[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

        input = self.qa_embed(qa)
        
        if state_in is None:
            state_in = (h_0, c_0)
        
        if self.cell_type.lower() == "lstm":
            state, state_out = self.rnn(input, state_in)
            output = self.fc(state)
            return output, state_out
        elif self.cell_type.lower() == "gru":
            state, state_out = self.rnn(input, h_0)
            output = self.fc(state)
            return output, state_out
        elif self.cell_type.lower() == "rnn":
            state, state_out = self.rnn(input, h_0)
            output = self.fc(state)
            return output, state_out 
 




