import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms

# to debug
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.adam import Adam
from src.utils.linear import Linear_np
from src.utils.relu import Relu_np
from src.utils.sigmoid import Sigmoid_np
from src.utils.ce import Cross_Entropy_np

class ModelAdam():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
        self.optimizer = Adam()
        
        self.linear_1 = Linear_np(input_channel , 256)
        self.linear_2 = Linear_np(256, 128)
        self.linear_3 = Linear_np(128, output_channel)
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.sigmoid = Sigmoid_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):
        
        #make x flatten [# of batch, 28*28 ]
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)
        
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        
        return x
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    def backward(self):
        d_prev = 1
        d_prev = self.criterion.backward(d_prev)
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear_3.backward(d_prev)
        d_prev = self.activation_2.backward(d_prev)
        d_prev = self.linear_2.backward(d_prev)
        d_prev = self.activation_1.backward(d_prev)
        d_prev = self.linear_1.backward(d_prev)
    
    def update_grad(self, learning_rate, batch_size):
        self.optimizer.update_grad('linear_3',self.linear_3,learning_rate/batch_size)
        self.optimizer.update_grad('linear_2',self.linear_2,learning_rate/batch_size)
        self.optimizer.update_grad('linear_1',self.linear_1,learning_rate/batch_size)
        
        self.optimizer.step()


def load_data():
    # MNIST 데이터셋 로드
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 데이터셋 일부만 사용
    train_dataset.data = train_dataset.data[:50000]
    train_dataset.targets = train_dataset.targets[:50000]

    test_dataset.data = test_dataset.data[:10000]
    test_dataset.targets = test_dataset.targets[:10000]

    # 데이터 로더
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=48, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader,test_loader):
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for images, labels in tqdm_batch:
            images, labels = images.numpy(), labels.numpy()

            output = model.forward(images)
            loss = model.loss(output,labels)
            
            epoch_loss+=loss
            
            model.backward()
            model.update_grad(LR,BATCH_SIZE)
            tqdm_batch.set_postfix(loss=epoch_loss / (len(tqdm_batch)))

        accuracy = validate(model, test_loader)
    return accuracy


def validate(model, test_loader):
    total_data = 0
    total_correct = 0
    with torch.no_grad():
        tqdm_batch = tqdm(test_loader, desc="Validation")
        for images, labels in tqdm_batch:
            # torch to numpy
            images, labels = images.numpy(), labels.numpy()
            output = model.forward(images)
            
            predicted_classes = np.argmax(output, axis=1)
            total_correct += (predicted_classes == labels).sum().item()
            total_data += labels.shape[0]
            tqdm_batch.set_postfix(accuracy=total_correct / total_data)
            
    accuracy = total_correct / total_data
    return accuracy


def start(train_Adam_np=False):
    print("loading...")
    train_loader, test_loader = load_data()
    print("========================\n\n")
    
    if train_Adam_np:
        print("========================")
        print("training numpy model with adam...")
        model = ModelAdam()
        accuracy = train(model, train_loader, test_loader)
        print(f"training with adam accuracy: {accuracy * 100:.2f}%")
        print("========================")


if __name__ == "__main__":
    random.seed(71)
    TOTAL_EPOCH = 5
    BATCH_SIZE = 48
    LR = 0.001
    start(train_Adam_np=True)