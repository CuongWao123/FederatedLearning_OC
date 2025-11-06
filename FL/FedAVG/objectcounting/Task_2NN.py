"""ObjectCounting: A Flower / PyTorch app for MNIST 2NN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, ShardPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net2NN(nn.Module):
    """MNIST 2NN: Multilayer-perceptron with 2 hidden layers.
    
    Architecture (from paper):
    - Input: 28x28 = 784 pixels
    - Hidden Layer 1: 200 units, ReLU activation
    - Hidden Layer 2: 200 units, ReLU activation
    - Output: 10 classes (digits 0-9)
    
    Total parameters: 199,210
    """

    def __init__(self, input_dim=784, hidden_dim=200, output_dim=10):
        super(Net2NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # 784 * 200 + 200 = 157,000
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 200 * 200 + 200 = 40,200
        self.fc3 = nn.Linear(hidden_dim, output_dim)     # 200 * 10 + 10 = 2,010
        # Total: 199,210 parameters

    def forward(self, x):
        # Flatten the input if needed (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # Hidden layer 1 with ReLU
        x = F.relu(self.fc1(x))
        
        # Hidden layer 2 with ReLU
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x


fds = None  # Cache FederatedDataset

# MNIST normalization: mean=0.1307, std=0.3081 (standard MNIST statistics)
pytorch_transforms = Compose([
    ToTensor(), 
    Normalize((0.1307,), (0.3081,))
])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


from datasets import disable_caching  # HF Datasets
import os, tempfile

def load_data(partition_id: int, num_partitions: int = 100, iid: bool = True, batch_size: int = 10):
    global fds

    disable_caching()  

    run_cache_dir = os.path.join(
        tempfile.gettempdir(), f"hf_ds_cache_{os.getpid()}_{partition_id}"
    )
    os.makedirs(run_cache_dir, exist_ok=True)

    if fds is None:
        if iid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            # Non-IID "pathological": 200 shards x 300, 2 shards/client -> 100 clients
            partitioner = ShardPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                num_shards_per_partition=2,
                shard_size=300,
                seed=42,
            )

        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
            cache_dir=run_cache_dir,   
        )

    partition = fds.load_partition(partition_id)  
    
    
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    if batch_size <= 0:
        batch_size = len(partition_train_test["train"])
        print(f"Using full batch mode: B=∞ (batch_size={batch_size})")

    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(partition_train_test["test"],  batch_size=32, shuffle=False)
    return trainloader, testloader

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set.
    
    Args:
        epochs: Number of local epochs (E in paper)
        lr: Learning rate
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Use SGD optimizer as in the paper (not Adam)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        running_loss += epoch_loss / len(trainloader)
    
    avg_trainloss = running_loss / epochs
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    net.eval()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)
    
    return avg_loss, accuracy


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)