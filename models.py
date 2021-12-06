import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# https://arxiv.org/pdf/1508.04025.pdf

class ModelV1(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout, bidirectional=True)
        

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        loss = F.mse_loss(0, 0)

        return loss

