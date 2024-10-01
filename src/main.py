import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms

from src.Adam import Adam

class ModelAdam(nn.Module):
    def __init__(self, input_size=28*28, output_size=10) -> None:
        super(ModelAdam, self).__init__()
        self.linear_1 = nn.Linear(input_size, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, output_size)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten
        x = self.activation_1(self.linear_1(x))
        x = self.activation_2(self.linear_2(x))
        x = self.linear_3(x)
        x= self.sigmoid(x)
        return x
    
    def loss(self, x, y):
        return self.criterion(x, y)
    
    def backward(self, output, target):
        loss = self.loss(output, target)
        loss.backward()

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

    # slice data
    train_dataset.data = train_dataset.data[:50000]
    train_dataset.targets = train_dataset.targets[:50000]

    test_dataset.data = test_dataset.data[:10000]
    test_dataset.targets = test_dataset.targets[:10000]
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=48, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader,test_loader, device):
    print('traning to:', device)
    model.train()
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for images, labels in tqdm_batch:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            model.backward(output, labels)

            model.update_grad(LR, BATCH_SIZE)
            epoch_loss += model.loss(output, labels).item()
            tqdm_batch.set_postfix(loss=epoch_loss / (len(tqdm_batch)))

        accuracy = validate(model, test_loader, device)
    return accuracy


def validate(model, test_loader, device):
    model.eval()
    total_data = 0
    total_correct = 0
    with torch.no_grad():
        tqdm_batch = tqdm(test_loader, desc="Validation")
        for images, labels in tqdm_batch:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predicted_classes = torch.argmax(output, axis=1)
            total_correct += (predicted_classes == labels).sum().item()
            total_data += labels.size(0)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        accuracy = train(model, train_loader, test_loader, device)
        print(f"training with adam accuracy: {accuracy * 100:.2f}%")
        print("========================")


if __name__ == "__main__":
    random.seed(71)
    TOTAL_EPOCH = 5
    BATCH_SIZE = 48
    LR = 0.001
    start(train_Adam_np=True)