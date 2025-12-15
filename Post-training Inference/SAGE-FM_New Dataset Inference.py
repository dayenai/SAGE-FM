"""
Xianghao Zhan, 12/14/2025
This code trains a foundation model for predicting masked gene expression features.
The model is trained on subgraphs of the Visium dataset.
The model is a Graph Neural Network (GNN) with a masked autoencoder (MAE) architecture.
This version of code load the pre-trained model and generate the embedding of a hold-out test set and also evaluate the masked gene expression efficacy.
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
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def nanstd(tensor, dim=None, keepdim=False):
    # Mask out NaN values
    mask = ~torch.isnan(tensor)
    valid_elements = tensor[mask]
    
    # Handle cases where the tensor is completely NaN along the dimension
    if dim is not None:
        mean = torch.nanmean(tensor, dim=dim, keepdim=True)
        std = torch.sqrt(torch.nanmean((tensor - mean)**2, dim=dim, keepdim=keepdim))
    else:
        std = torch.sqrt(torch.nanmean((tensor - torch.nanmean(tensor))**2))
    return std

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

def calculate_metrics(predictions, targets, mask):
    """
    Calculate MSE, RMSE, R2 and Pearson correlationship metrics for the masked elements.

    Parameters:
    - predictions: Tensor of model predictions.
    - targets: Tensor of ground truth values.
    - mask: Binary mask tensor indicating which elements are masked (1 for masked, 0 otherwise).

    Returns:
    - metrics: Dictionary containing MSE, RMSE, and R2 values.
    """

    mask = mask.to("cpu") # (Num_graphs * 15 neighbors) X Num_gene_features
    targets = targets.to("cpu") # (Num_graphs * 15 neighbors) X Num_gene_features

    # Calculate R2 for each sample
    sample_r2s = []
    for i in np.arange(0, mask.shape[0], 15): # 15 neighbors per central spot, only evaluate on the central spot
        target = targets[i, mask[i, :]]
        prediction = predictions[i, mask[i, :]]
        #print(target.numpy().shape, prediction.numpy().shape)
        total_variance = torch.var(target).item()

        if total_variance == 0:  # Handle edge case where variance is zero
            sample_r2s.append(float('nan'))
        else:
            explained_variance = torch.var(target - prediction).item()
            r2 = 1 - (explained_variance / total_variance)
            sample_r2s.append(r2)

    # Convert to tensor for easier computation
    sample_r2s = torch.tensor(sample_r2s)

    # Calculate summary statistics for R2 values
    r2_mean = torch.nanmean(sample_r2s).item()
    r2_median = torch.nanmedian(sample_r2s).item()
    r2_std = nanstd(sample_r2s).item()

    # Apply the mask
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    # Compute MSE
    mse = torch.mean((masked_predictions - masked_targets) ** 2).item()

    # Compute RMSE
    rmse = torch.sqrt(torch.tensor(mse)).item()


    # Return metrics
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2_mean": r2_mean,
        "R2_median": r2_median,
        'R2_std': r2_std
    }

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

class HoldOutTestDataGenerator(torch.utils.data.IterableDataset):
    """
    The DataGenerator class generates masked subgraphs on-the-fly for the hold-out test set.
    This class is used to evaluate the performance of the pre-trained model and get embeddings for the test samples.
    The test set is not used for training the model.
    """
    def __init__(self, data_dir, dataset, mask_percentage=0.3, seed=9001):
        self.data_dir = data_dir # Directory containing the subgraph data
        self.dataset = dataset # Name of the dataset: GBM/HEST1k
        self.mask_percentage = mask_percentage
        self.files = [f for f in os.listdir(data_dir) if f.endswith(f"_{dataset}.pt")]

    def __iter__(self):
        for file in self.files:
            subgraphs = torch.load(os.path.join(self.data_dir, file)) # A list of subgraphs (Data type): (number of spots), each subgraph has (data.x, data.edge, shape is (15, 14558))
            labels = [file.split(f"_{self.dataset}.pt")[0]] * len(subgraphs) # Number of central spots
            for subgraph, label in zip(subgraphs, labels):
                subgraph.y = label # One label per subgraph for the central spot
                subgraph = generate_central_mask(subgraph, mask_percentage=self.mask_percentage)
                yield subgraph


# Function to load a pre-trained model
def load_model(checkpoint_path, dataset):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Extract the saved parameters
    hidden_channels = checkpoint['hidden_channels']
    learning_rate = checkpoint['learning_rate']
    batch_size = checkpoint['batch_size']
    
    # Define the model structure
    try:
        in_channels = checkpoint['in_channels']  # Update this if needed
    except KeyError:
        if dataset == 'HESTk':
            in_channels = 14558  # Default value for v4 models
        else:
            in_channels = 14558
    try:    
        num_gene_features = checkpoint['num_gene_features']  # Update this if needed
    except KeyError:
        if dataset == 'HEST1k':
            num_gene_features = 14558  # Default value for v4 models
        else:
            num_gene_features = 14558

    model = PretrainModel(in_channels, hidden_channels, num_gene_features)
    
    # Load the state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint

def evaluate_get_embedding_with_generator(
    model_name, 
    test_dataset, 
    device=None):

    # 1. Define the specifications for the model
    print('Loading model...', model_name)
    model_dir  = "./HEST1k_selected_models"
    model_path = os.path.join(model_dir, model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dataset = str(model_name.split("_")[0])
    test_dataset = str(test_dataset)
    mask_percentage = float(model_name.split("mask_")[1].split('_')[0])
    batch_size = int(model_name.split("batch_")[1].split('.pt')[0])

    # 2. Load the model and define the data generators
    model, optimizer, scheduler, checkpoint = load_model(model_path, dataset=model_dataset)

    data_dir_dict = {
        'GBM': "./gnn_subgraphs/GBM", # Change the directory into the subgraphs after running the data preprocessing
    }
    data_dir = data_dir_dict[test_dataset]

    # Create data generators with zero masking to acquire the subgraph embeddings
    test_dataset = HoldOutTestDataGenerator(data_dir=data_dir, dataset=test_dataset, mask_percentage=0)
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 3. Save the embeddings under zero masking
    model = model.to(device)
    model.eval()
    y_test_embeddings = []
    y_test_label = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, embeddings = model(data)
            y_test_embeddings.append(embeddings.cpu()[::15]) # Embeddings for the central spot's gene expression features (hidden state) (only central spots)
            y_test_label.append(data.y) # Spot labels (only central spots)

    y_test_embeddings = torch.cat(y_test_embeddings, dim=0)
    y_test_label = np.concatenate(y_test_label, axis=0)
    print('Test Embeddings Shape:', y_test_embeddings.size())
    print('Test Labels Shape:', y_test_label.shape)

    os.makedirs('./HEST1k_holdouttest_subgraph_embeddings', exist_ok=True)
    np.save(f"./HEST1k_holdouttest_subgraph_embeddings/{model_name.split('.pt')[0]}_subgraph_embedding.npy", y_test_embeddings.numpy())
    np.save(f"./HEST1k_holdouttest_subgraph_embeddings/{model_name.split('.pt')[0]}_subgraph_labels.npy", np.array(y_test_label))

    del test_dataset, test_loader, y_test_embeddings, y_test_label

    # 4. Evaluate the Test Performance by masking the central spot and making prediction with the pre-trained GNN model
    test_dataset = HoldOutTestDataGenerator(data_dir=data_dir, dataset=test_dataset, mask_percentage=mask_percentage)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = model.to(device)
    model.eval()
    test_loss = 0
    num_test_batches = 0
    y_test = []
    y_test_pred = []
    y_test_mask = []
    y_test_masked = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            predictions, embeddings = model(data)
            y_test.append(data.original_x.cpu()[::15]) # Original gene expression features (only central spots)
            y_test_pred.append(predictions.cpu()[::15]) # Predicted gene expression features (only central spots)
            y_test_mask.append(data.mask.cpu()[::15]) # Mask for the central spot's gene expression features (only central spots)
            y_test_masked.append(data.x.cpu()[::15]) # Masked gene expression features (only central spots)
            test_loss += masked_mse_loss(predictions, data.original_x, data.mask).item()
            num_test_batches += 1

    test_loss /= num_test_batches
    print(f"Test Loss: {test_loss:.4f}")

    y_test = torch.cat(y_test, dim=0)
    y_test_pred = torch.cat(y_test_pred, dim=0)
    y_test_mask = torch.cat(y_test_mask, dim=0)
    y_test_masked = torch.cat(y_test_masked, dim=0)
    print('Y_Test Shape: ', y_test.size())
    print('Y_Pred Shape: ', y_test_pred.size())

    test_metrics = calculate_metrics(y_test_pred, y_test, y_test_mask)
    test_metrics_baseline = calculate_metrics(y_test_masked, y_test, y_test_mask)

    print(f"Test Set Metrics: MSE: {test_metrics['MSE']:.6f}, RMSE: {test_metrics['RMSE']:.4f}, R2_mean: {test_metrics['R2_mean']:.4f}, R2_median: {test_metrics['R2_median']:.4f}, R2_std: {test_metrics['R2_std']:.4f}")
    print(f"Test Set Metrics Baseline: MSE: {test_metrics_baseline['MSE']:.6f}, RMSE: {test_metrics_baseline['RMSE']:.4f}, R2_mean: {test_metrics_baseline['R2_mean']:.4f}, R2_median: {test_metrics_baseline['R2_median']:.4f}, R2_std: {test_metrics_baseline['R2_std']:.4f}")

    os.makedirs('./HEST1k_holdouttest_subgraph_embeddings', exist_ok=True)
    np.save(f"./HEST1k_holdouttest_subgraph_embeddings/{model_name.split('.pt')[0]}_subgraph_gt.npy", y_test.numpy())
    np.save(f"./HEST1k_holdouttest_subgraph_embeddings/{model_name.split('.pt')[0]}_subgraph_prediction.npy", y_test_pred.numpy())
    np.save(f"./HEST1k_holdouttest_subgraph_embeddings/{model_name.split('.pt')[0]}_subgraph_mask.npy", y_test_mask.numpy())

    log_memory_usage("After testing")

    del y_test, y_test_pred, y_test_mask, y_test_masked, test_metrics, test_metrics_baseline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a foundation model for predicting masked gene expression features.")
    parser.add_argument('--model_name', type=str, default='HEST1k_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt', help='Name of Pre-trained Model')
    parser.add_argument('--test_dataset', type=str, default='GBM', help='Name of the test data')
    args = parser.parse_args()

    # Run the evaluation pipeline
    evaluate_get_embedding_with_generator(args.model_name, args.test_dataset)

if __name__ == "__main__":
    main()
