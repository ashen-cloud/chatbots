#!/opt/homebrew/bin/python3

from models import ModelV1
from preprocess import create_dataloader, MAX_LENGTH

from tqdm import tqdm

import torch
import torch.nn as nn

dataloader, text_transform, voc = create_dataloader()

input_size = next(iter(dataloader))[0].shape[1]

model = ModelV1(input_size, len(voc), MAX_LENGTH)

dec_optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-7)
dec_loss_fn = nn.L1Loss()

EPOCHS = 10

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader)):
        x, y = data[0], data[1]

        optim.zero_grad()

        out = model(x)

        loss = dec_loss_fn(out, y)
        print('loss', loss)

        loss.backward()
        dec_optim.step()

        running_loss += loss.item()

    print('epoch', epoch, 'loss', running_loss / len(dataloader))
