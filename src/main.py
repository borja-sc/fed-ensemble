import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from FedEnsemble import fed_ensemble
from models import BaseModel

def main():
    # Set Fed-Ensemble hyperparameters
    NUM_MODELS = 5
    NUM_CLIENTS = 100
    NUM_STRATA = 10
    NUM_SELECTED_CLIENTS = 5
    NUM_AGES = 50
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001

    hidden_size = 100
    num_classes = 10

    # Transformations applied to the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR-10 training dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Load the CIFAR-10 validation dataset
    validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create model architecture
    network = BaseModel(hidden_size, num_classes)

    # Perform Fed-Ensemble in federated learning
    train(NUM_CLIENTS, NUM_MODELS, NUM_SELECTED_CLIENTS, NUM_AGES, BaseModel, train_dataset,
                                   validation_dataset, batch_size, num_epochs, learning_rate, NUM_STRATA)

if __name__ == '__main__':
    main()
