#!/opt/homebrew/bin/python3

from models import ModelV1
from preprocess import create_dataloader

from tqdm import tqdm

import torch
import torch.nn as nn

# todo: save/load, wrapper, improve nn arch (use transformer or conv)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dataloader, text_transform, voc = create_dataloader()

input_size = next(iter(dataloader))[0].shape[1]

model = ModelV1(input_size, vocab_len=len(voc), out_dim=input_size)
model.to(device)

dec_optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-7)
dec_loss_fn = nn.L1Loss()

EPOCHS = 10

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader)):
        x, y = data[0].to(device), data[1].to(device)

        dec_optim.zero_grad()

        out, _ = model(x, y)

        loss = dec_loss_fn(out, y)

        loss.backward()
        dec_optim.step()

        running_loss += loss.item()

    print('epoch', epoch, 'loss', running_loss / len(dataloader))
