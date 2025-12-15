"""
Author: Xianghao Zhan, 12/13/2025
This code trains a foundation model for predicting masked gene expression features.
The model is trained on subgraphs of the Visium dataset.
The model is a Graph Neural Network (GNN) with a masked autoencoder (MAE) architecture.
This version of code uses a generator to generate masked subgraphs on-the-fly.
"""

# Load required libraries
import torch
import psutil
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import warnings
import random
import argparse

# Ignore warnings and set random seed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.random.manual_seed(9001)
np.random.seed(9001)
random.seed(9001)

# Define utility functions
def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{stage}] Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")


def masked_mse_loss(predictions, targets, mask):
    """
    Computes the Mean Squared Error (MSE) loss for masked elements only.

    Parameters:
    - predictions: Model predictions.
    - targets: Ground truth values.
    - mask: Binary mask indicating which values are masked.

    Returns:
    - MSE loss for the masked elements.
    """
    mask = mask.to(predictions.device)
    loss = ((predictions - targets) ** 2) * mask  # Retain loss only for masked elements
    return loss.sum() / mask.sum()  # Normalize by the number of masked elements


class PretrainModel(torch.nn.Module):
    """
    The PretrainModel class defines the architecture of the foundation model.
    """
    def __init__(self, in_channels, hidden_channels, num_gene_features):
        """
        Initializes the PretrainModel class.
        parameters:
        - in_channels: The number of input channels (the dimensionality of node representation).
        - hidden_channels: A list of hidden layer sizes.
        - num_gene_features: The number of gene expression features (should be the same as the dimensionality of node representation).
        """
        super(PretrainModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GCNConv(hidden_channels[2], hidden_channels[3])
        self.conv5 = GCNConv(hidden_channels[3], hidden_channels[4])
        self.fc_gene = torch.nn.Linear(hidden_channels[-1], num_gene_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        hidden_state = self.conv5(x, edge_index)
        gene_predictions = torch.relu(self.fc_gene(hidden_state)) # ReLU activation for non-negative gene expression values (after np.log1p normalization)
        return gene_predictions, hidden_state


def generate_central_mask(data, mask_percentage=0.3):
    """
    Generate a mask for the central spot's gene expression features and retain the original data as ground truth.

    Parameters:
    - data: The input subgraph data (including node features, edge indices, etc.).
    - mask_percentage: The percentage of gene expression features to mask.

    Returns:
    - data: The masked subgraph data with the original gene expression features retained (adding the masked values, the mask and original data).
    """
    num_cells, num_gene_features = data.x.shape

    # Generate a random mask for the central spot's gene expression features
    central_mask = torch.rand(num_gene_features) < mask_percentage

    # Apply the mask to the central spot's gene expression features
    original_x = data.x.clone()
    masked_x = data.x.clone()
    masked_x[0, central_mask] = 0 # Mask the central spot's gene expression features

    # Store the mask and original data in the data object
    mask = torch.zeros_like(data.x, dtype=torch.bool)
    mask[0, :] = central_mask
    data.x = masked_x
    data.mask = mask
    data.original_x = original_x
    return data

class DataGenerator(torch.utils.data.IterableDataset):
    """
    The DataGenerator class generates masked subgraphs on-the-fly for training the foundation model.
    """
    def __init__(self, data_dir, dataset, split, mask_percentage=0.3, seed=9001):
        self.data_dir = data_dir # Directory containing the subgraph data
        self.dataset = dataset # Dataset used to train: HEST1k
        self.split = split # Split of the dataset: train, val, or test (80%, 10%, 10%, partition by each sample's subgraphs)
        self.mask_percentage = mask_percentage
        self.files = [f for f in os.listdir(data_dir) if f.endswith(f"_{dataset}.pt")]
        random.seed(seed)
        random.shuffle(self.files)
        total = len(self.files)
        if split == 'train':
            self.files = self.files[:int(0.8 * total)]
        elif split == 'val':
            self.files = self.files[int(0.8 * total):int(0.9 * total)]
        elif split == 'test':
            self.files = self.files[int(0.9 * total):]

    def __iter__(self):
        for file in self.files:
            subgraphs = torch.load(os.path.join(self.data_dir, file))
            for subgraph in subgraphs:
                yield generate_central_mask(subgraph, self.mask_percentage)


def train_pipeline_with_generator(
    data_dir,
    dataset,
    in_channels,
    hidden_channels,
    num_gene_features,
    mask_percentage=0.3,
    batch_size=32,
    epochs=500,
    lr=0.0003,
    weight_decay=1e-4,
    patience=10,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Create data generators
    train_dataset = DataGenerator(data_dir, dataset, 'train', mask_percentage)
    val_dataset = DataGenerator(data_dir, dataset, 'val', mask_percentage)
    test_dataset = DataGenerator(data_dir, dataset, 'test', mask_percentage)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and learning rate scheduler
    model = PretrainModel(in_channels, hidden_channels, num_gene_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=0.9) # Reduce learning rate on plateau, if validation loss does not improve, reduce learning rate by a factor of 0.5

    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0
        num_batches = 0

        for data in train_loader:
            data = data.to(device) # Move data to device
            optimizer.zero_grad() # Zero the gradients
            predictions, _ = model(data) # Forward pass
            loss = masked_mse_loss(predictions, data.original_x, data.mask) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update model parameters
            train_loss += loss.item()
            num_batches += 1
        
        # Average training loss
        train_loss /= num_batches

        val_loss = 0
        num_val_batches = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                predictions, _ = model(data)
                val_loss += masked_mse_loss(predictions, data.original_x, data.mask).item()
                num_val_batches += 1

        val_loss /= num_val_batches
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'hidden_channels': hidden_channels,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'mask_percentage': mask_percentage,
                'batch_size': batch_size}, os.path.join('./models', f"{dataset}_best_model_epoch_{epoch}_hidden_{'_'.join(map(str, hidden_channels))}_mask_percentage_{mask_percentage}_lr_{lr}.pt"))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)

    model.eval()
    test_loss = 0
    num_test_batches = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            predictions, _ = model(data)
            test_loss += masked_mse_loss(predictions, data.original_x, data.mask).item()
            num_test_batches += 1

    test_loss /= num_test_batches
    print(f"Test Loss: {test_loss:.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a foundation model for predicting masked gene expression features.")
    parser.add_argument('--dataset', type=str, default='HEST1k', choices=['HEST1k'], help='The dataset used for training.')
    parser.add_argument('--in_channels', type=int, default=1280, help='Number of input channels.')
    parser.add_argument('--hidden_channels', type=int, nargs='+', default=[1024, 512, 512, 512, 1024], help='List of hidden layer sizes.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_gene_features', type=int, default=1280, help='Number of gene features.')
    parser.add_argument('--mask_percentage', type=float, default=0.5, help='Percentage of masking for gene features.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1), help='Train/validation/test split ratios.')

    args = parser.parse_args()

    data_dir_dict = {
        'HEST1k': "./gnn_subgraphs/HEST1k", # You can add additional dataset you want to use to train SAGE-FM
    }
    data_dir = data_dir_dict[args.dataset]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train the model
    train_pipeline_with_generator(
        data_dir=data_dir,
        dataset=args.dataset,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_gene_features=args.num_gene_features,
        mask_percentage=args.mask_percentage,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device
    )

if __name__ == "__main__":
    main()
