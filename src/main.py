import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from FedEnsemble import fed_ensemble
from models import BaseModel, train_local

def main():
    # Set Fed-Ensemble hyperparameters
    NUM_MODELS = 5
    NUM_CLIENTS = 100
    NUM_SELECTED_CLIENTS = 10
    NUM_AGES = 5
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    
    input_size = 3 * 32 * 32  # CIFAR-10 image size
    hidden_size = 100
    num_classes = 10

    # Transformations applied to the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Load the CIFAR-10 validation dataset
    validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create model architecture
    network = BaseModel(input_size, hidden_size, num_classes)

    # Perform Fed-Ensemble in federated learning
    ensemble_models = fed_ensemble(NUM_CLIENTS, NUM_MODELS, NUM_SELECTED_CLIENTS, NUM_AGES, BaseModel, train_dataset, 
                                   validation_dataset, batch_size, num_epochs, learning_rate)

    # Save the ensemble model
    for model in ensemble_models:
      torch.save(model.state_dict(), f"model_{i}.pth")

if __name__ == '__main__':
    main()
