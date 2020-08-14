import torchvision.models as models
from torchvision.transforms import transforms
import torchvision
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import Subset, DataLoader
import sklearn.model_selection

from ignite.engine import Engine, Events
from ignite.metrics import Loss, Accuracy
import time

from torch.utils.tensorboard import SummaryWriter
from ignite.handlers import ModelCheckpoint

class Señalizador(nn.Module):
    def __init__(self, train_data, validation_data, test_data):
        super(type(self), self).__init__()
        self.conv1 = torch.nn.Conv2d(kernel_size = 7, in_channels = 3, out_channels = 18, bias = True)
        self.conv2 = torch.nn.Conv2d(kernel_size = 5, in_channels = 18, out_channels = 18, bias = True)
        self.conv3 = torch.nn.Conv2d(kernel_size = 3, in_channels = 18, out_channels = 36)
        self.conv4 = torch.nn.Conv2d(kernel_size = 3, in_channels = 36, out_channels = 64)
        self.conv5 = torch.nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 64, bias = True)
        self.conv6 = torch.nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128, bias = True)
        self.conv7 = torch.nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 254, padding = 1, bias = True)
        self.mpool = torch.nn.MaxPool2d(kernel_size = 2)
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(in_features = 254, out_features = 128)
        self.linear2 = torch.nn.Linear(in_features = 128, out_features = 16)
        self.linear3 = torch.nn.Linear(in_features = 16, out_features = 4)
        self.dropout = torch.nn.Dropout(p = 0.2)
        
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        
        self.device = torch.device('cuda:0')

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 5e-3)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
        self = self.to(self.device)
        
        self.loaders()
        
        
    def forward(self, x):
        x = self.mpool(self.activation(self.conv1(x)))
        x = self.mpool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.mpool(self.activation(self.conv4(x)))
        x = self.mpool(self.activation(self.conv5(x)))
        x = self.mpool(self.activation(self.conv6(x)))
        x = self.mpool(self.activation(self.conv7(x)))
        x = x.view(-1, self.linear1.in_features)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def loaders(self):
        self.train_loader = DataLoader(self.train_data, shuffle = True, batch_size = 32)
        self.valid_loader = DataLoader(self.validation_data, shuffle = False, batch_size = 256)
        self.test_loader = DataLoader(self.test_data, shuffle = False, batch_size = 512)
        print("Loaders initialized")

    def load_checkpoint(self, dir):
        self.load_state_dict(torch.load(dir))
        
    def train_one_step(self, engine, batch):
        self.optimizer.zero_grad()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        yhat = self.forward(x)
        loss = self.criterion(yhat, y)
        loss.backward()
        self.optimizer.step()
        del x
        del y
        torch.cuda.empty_cache()
        return loss.item() # Este output puede llamar luego como trainer.state.output
        
    def evaluate_one_step(self, engine, batch):
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.forward(x)
            del x
            loss = self.criterion(yhat, y)
            torch.cuda.empty_cache()
            return yhat, y
    
    def train_epochs(self, max_epochs):
        self.trainer = Engine(self.train_one_step)
        self.evaluator = Engine(self.evaluate_one_step)
        self.metrics = {'Loss': Loss(self.criterion), 'Acc': Accuracy()}
        for name, metric in self.metrics.items():
            metric.attach(self.evaluator, name)
            
        with SummaryWriter(log_dir="/tmp/tensorboard/Transform"+ str(type(self))[17:len(str(type(self)))-2]) as writer:
            @self.trainer.on(Events.EPOCH_COMPLETED(every=1)) # Cada 1 epocas
            def log_results(engine):
                # Evaluo el conjunto de entrenamiento
                self.eval()
                self.evaluator.run(self.train_loader) 
                writer.add_scalar("train/loss", self.evaluator.state.metrics['Loss'], engine.state.epoch)
                writer.add_scalar("train/accy", self.evaluator.state.metrics['Acc'], engine.state.epoch)

                # Evaluo el conjunto de validación
                self.evaluator.run(self.valid_loader)
                writer.add_scalar("valid/loss", self.evaluator.state.metrics['Loss'], engine.state.epoch)
                writer.add_scalar("valid/accy", self.evaluator.state.metrics['Acc'], engine.state.epoch)
                self.train()
            # Guardo el mejor modelo en validación
            best_model_handler = ModelCheckpoint(dirname='.', require_empty=False, filename_prefix="best", n_saved=1,
                                                 score_function=lambda engine: -engine.state.metrics['Loss'],
                                                 score_name="val_loss")
            # Lo siguiente se ejecuta cada ves que termine el loop de validación
            self.evaluator.add_event_handler(Events.COMPLETED, 
                                    best_model_handler, {f'Transform{str(type(self))[17:len(str(type(self)))-2]}': model})
        
        self.trainer.run(self.train_loader, max_epochs=max_epochs)
        
    def test(self, confussion, report):
        self.eval()
        test_targets = np.array(self.test_data.targets)
        prediction_test =[]
        for mbdata, label in self.test_loader:
            mbdata = mbdata.to(self.device)
            logits = self.forward(mbdata).to("cpu")
            prediction_test.append(logits.argmax(dim=1).detach().numpy())
            del mbdata
            del logits
            torch.cuda.empty_cache()
        prediction_test = np.concatenate(prediction_test)
        cm = confusion_matrix(test_targets, prediction_test)
        if(confussion):
            display(cm)
        if(report):
            print(classification_report(test_targets, prediction_test))
        self.train()
        return cm