import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

from dataloader import ImageDataset
from model import SRCNN

np.random.seed(69)
torch.manual_seed(69)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using: {device}')
# if str(device) == 'cuda': print(torch.cuda.get_device_name()) 

BATCH_SIZE = 2
EPOCHS = 20

train_set = ImageDataset("data/train/", 2)
test_set = ImageDataset("data/test/", 2)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

# init model
model = SRCNN()
model.to(device)

# loss function
loss_fn = nn.MSELoss()

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001)

all_losses = []
test_losses = []
for epoch in range(EPOCHS):
    t0 = time()
    all_losses.append([])
    for batch in train_loader:
        # load data to the device
        x, y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        out = model.forward(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        # hold the loss
        all_losses[-1].append(loss.item())

    with torch.no_grad():
        t1 = time()
        test_loss = 0
        for batch in test_loader:
            # load data to the device
            x, y = batch[0].to(device), batch[1].to(device)
            out = model.forward(x)
            loss = loss_fn(out, y)
            test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        print(f'{epoch}: Val loss: {test_losses[-1]} | loss: {sum(all_losses[-1]/len(train_loader))} | Train time: {t1-t0:2f} | Test time: {time()-t1:2f}')

    # TODO: save model here
