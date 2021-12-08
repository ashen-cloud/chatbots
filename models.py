import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import pad_token_ind

# todo: loss, architecture, refactor preprocessing? 

# https://arxiv.org/pdf/1508.04025.pdf -- attention


class ModelV1(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=0, vocab_len=0, vect_len=0):
        super().__init__()

        self.input_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_len, vect_len, max_norm=True)

        self.attn = nn.Linear(hidden_size, hidden_size)
             
        self.encoder = nn.Sequential(
            nn.GRU(input_size, hidden_size, num_layers, dropout, bidirectional=True)
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.GRU(input_size, hidden_size, num_layers, dropout, bidirectional=True)
        )
        

    def forward(self, x, init_state=None):
        embedded = self.embedding(x)
        print('embedded', embedded)

        encoder_out = self.encoder(embedded.unsqueeze(1))
        print('encoder_out', encoder_out)

        decoder_out = self.decoder(encoder_out)
        print('decoder_out', decoder_out)

        return out, hidden

