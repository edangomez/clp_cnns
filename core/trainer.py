import os
import numpy as np
import torch 
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
import tqdm


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_dir: Optional[str] = None,
        
        metric=None
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.train_dataset = train_loader
        self.eval_loader = eval_loader
        self.metric = metric
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_loss = 10000
        self.best_epoch = 0
        self.best_model = ""
        self.model_dir = model_dir
                
    def train(self):
        for epoch in range(self.num_epochs):
            
            self.model.train()

            for batch_idx, (data, targets) in enumerate(tqdm.tqdm(self.train_dataset)):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                # forward
                # TODO: Perform a forward pass of the model, calculate the loss and perform backpropagation
                scores = self.model(data)
                
                loss = self.criterion(scores, targets)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()
                
            # save best model
            mean_loss = torch.mean(loss)
            if mean_loss < self.best_loss:
                best_loss = mean_loss
                
                # TODO: create and save the state dict of the best model
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': best_loss}
                print(self.model_dir)
                if not self.model_dir:
                    path = os.makedirs(self.model_dir)
                    self.model_dir = path
                    
                torch.save(
                    state,
                    os.path.join(
                        self.model_dir,  # type: ignore
                        f'best_model_{epoch}.pth'
                    )  # type: ignore
                )