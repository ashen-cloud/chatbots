import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random

from preprocess import pad_token_ind

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# https://arxiv.org/pdf/1508.04025.pdf -- attention


class ModelV1(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=0, vocab_len=0, vect_len=0, out_dim=0):
        super(ModelV1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.out_dim = out_dim

        # self.attn = nn.Linear(hidden_size, hidden_size)

        self.encoder = nn.Sequential(
            nn.Embedding(vocab_len, input_size, max_norm=True),
            nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        )

        self.dec_embedding = nn.Embedding(vocab_len, input_size, max_norm=True)
        self.decoder = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.decoder_fc = nn.Linear(input_size * hidden_size, out_dim)


    def forward(self, x, y=None):
        _, encoder_state = self.encoder(x)

        x_embedded = self.dec_embedding(x)
        tf = random() > 0.5

        decoder_out, _ = self.decoder(self.dec_embedding(y) if tf and y != None else x_embedded, encoder_state)

        dec_reshaped = decoder_out.reshape(x.shape[0], self.hidden_size * self.input_size)

        norm_dec_out = self.decoder_fc(dec_reshaped)

        return norm_dec_out, y

