import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class ModelV1(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        loss = F.mse_loss(0, 0)

        return loss

