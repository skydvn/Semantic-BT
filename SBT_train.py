from functools import partial
from typing import Sequence, Tuple, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet34
from torchvision.utils import make_grid
from torchvision import models
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

logger = CSVLogger("logs_out", name="encoder_logs")

batch_size = 64
num_workers = 8
max_epochs = 200
z_dim = 2048


train_transform = BarlowTwinsTransform(
    train=True, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization()
)
train_dataset = CIFAR10(root=".", train=True, download=True, transform=train_transform)

val_transform = BarlowTwinsTransform(
    train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization()
)
val_dataset = CIFAR10(root=".", train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True,
                        pin_memory=True)

encoder = resnet34()
encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

# replace classification fc layer of Resnet to obtain representations from the backbone
encoder.fc = nn.Identity()

encoder_out_dim = 512

model = BarlowTwins(
    encoder=encoder,
    encoder_out_dim=encoder_out_dim,
    num_training_samples=len(train_dataset),
    batch_size=batch_size,
    z_dim=z_dim,
)

checkpoint_callback = ModelCheckpoint(every_n_epochs=100, save_top_k=-1, save_last=True)

trainer = L.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices=1,
    logger=logger
)

trainer.fit(model, train_loader, val_loader)


class classifire(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        return x


classifire_model = classifire()

'''the below methods are the 2 implemented optimizer and schedulers 
we can shift between them '''
ai_optimizer = torch.optim.Adam(classifire_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ai_optimizer, 200)
goai_loss_fn = nn.CrossEntropyLoss()
# ai_optimizer = torch.optim.SGD(classifire_model.parameters(), lr=1e-3,
#                                 momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ai_optimizer, 200)


running_loss = 0.0

running_corrects = 0
epochs = 100

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    running_corrects = 0
    for batch in train_loader:
        (img1, img2, _), label = batch
        img1 = img1.to('cuda')
        model = model.to('cuda')
        label = label.to('cuda')
        model.eval()
        classifire_model.to('cuda')
        classifire_model.train()

        encoder_out = model.forward(img1)
        input_to_goai = torch.unsqueeze(encoder_out,
                                        dim=1)  ### these unsuqeese is done to make the data follow NCHW pattern which is used inside the pytorch as a data format
        input_to_goai = torch.unsqueeze(input_to_goai, dim=1)
        # y_pred = GOAI_model(encoder_out)

        # GOAI_model.to('cuda')
        y_pred = classifire_model(input_to_goai)
        _, preds = torch.max(y_pred, 1)

        loss = goai_loss_fn(y_pred, label)
        ai_optimizer.zero_grad()

        loss.backward()

        ai_optimizer.step()
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == label.data)
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

correct = 0
total = 0
with torch.inference_mode():
    for batch in val_loader:
        (img1, img2, _), label = batch
        model.eval()
        classifire_model.eval()
        model.to('cpu')
        classifire_model.to('cpu')

        encoder_out = model.forward(img1)
        input_to_goai = torch.unsqueeze(encoder_out,
                                        dim=1)  ### these unsuqeese is done to make the data follow NCHW pattern which is used inside the pytorch as a data format
        input_to_goai = torch.unsqueeze(input_to_goai, dim=1)
        y_pred = classifire_model(input_to_goai)
        _, preds = torch.max(y_pred.data, 1)
        total += label.size(0)
        correct += (preds == label).sum().item()
# epoch_acc = running_corrects.double() / len(val_loader.dataset)
# print('Acc: {:.4f}'.format(epoch_acc))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))