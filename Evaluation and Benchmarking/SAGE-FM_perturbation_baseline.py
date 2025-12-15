"""
Xianghao Zhan, 06/16/2025
BASELINE CONTROL VERSION: This code uses a pre-trained spatial foundation model for in silico perturbation study.
This version uses RANDOMLY SELECTED BASELINE GENES instead of downstream genes for comparison.
This allows comparison of effect sizes between actual downstream genes and random baseline genes.

The script uses the same input genes (ligands) for perturbation but predicts randomly selected genes
instead of the known downstream genes, serving as a control to assess whether the perturbation effects
are specific to downstream genes or general model behavior.
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
import pandas as pd
import argparse

print("hello! (Baseline Control Version)")

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

def load_csv(file_path):
    """
    This function loads a CSV file containing paired genes and their indices, min, and max values. Convert it into a dictionary for easy access.
    The CSV file should have columns: 'Input Gene', 'Output Gene', 'Input ID', 'Output ID, '0th' (minimum value of the input gene expression), '100th' (maximum value of input gene expression).
    This function groups all output genes for each input gene (one input gene can have multiple downstream output genes).
    """
    df = pd.read_csv(file_path)
    input_output_dict = {}
    all_output_ids = np.array(df["Output ID"].tolist())  # Numpy array of all output indices
    
    # Group all output genes by input gene
    for input_gene in df['Input Gene'].unique():
        input_gene_rows = df[df['Input Gene'] == input_gene]
        
        # Get input gene information (should be the same for all rows with same input gene)
        input_index = int(input_gene_rows.iloc[0]['Input ID'])
        min_value = float(input_gene_rows.iloc[0]['0th'])
        max_value = float(input_gene_rows.iloc[0]['100th'])
        
        # Collect all output genes and their indices for this input gene
        output_genes = input_gene_rows['Output Gene'].tolist()
        output_indices = np.array(input_gene_rows['Output ID'].tolist(), dtype=int)
        
        # Get unmatched genes (all output genes that are NOT downstream of this input gene)
        unmatched_indices = np.array([idx for idx in all_output_ids if idx not in output_indices], dtype=int)
        
        # Store the information in a dictionary
        input_output_dict[input_gene] = {
            'output_genes': output_genes,  # List of all output gene names
            'output_indices': output_indices,  # Array of all output gene indices for this input gene
            'unmatched_indices': unmatched_indices,  # Array of unmatched output gene indices
            'input_index': input_index,
            'min_value': min_value,
            'max_value': max_value
        }
    
    return input_output_dict, all_output_ids


def select_random_baseline_genes(num_genes_to_select, exclude_indices, total_num_genes, seed=None):
    """
    Randomly select baseline genes for comparison.
    
    Parameters:
    - num_genes_to_select: Number of genes to randomly select (should match number of downstream genes)
    - exclude_indices: Array of gene indices to exclude (input gene + downstream genes)
    - total_num_genes: Total number of genes in the dataset (e.g., 14558 for v4)
    - seed: Random seed for reproducibility
    
    Returns:
    - baseline_indices: Array of randomly selected gene indices
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create set of all possible gene indices
    all_gene_indices = set(range(total_num_genes))
    
    # Exclude input gene and downstream genes
    exclude_set = set(exclude_indices)
    available_indices = np.array(list(all_gene_indices - exclude_set))
    
    # Check if we have enough genes to select
    if len(available_indices) < num_genes_to_select:
        raise ValueError(f"Not enough genes available. Need {num_genes_to_select}, but only {len(available_indices)} available after exclusions.")
    
    # Randomly select genes
    selected_indices = np.random.choice(available_indices, size=num_genes_to_select, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort for consistency
    
    return selected_indices


def insilico_perturbation(data, input_output_dict, input_gene, output_ids, perturb_level = 0):
    """
    Mask the central spot's gene expression features based on the input gene and all output genes of interest.
    This function will set the input gene expression levels of the neighboring spot to be 5 values between the min and max values of the input gene expression levels.
    
    Parameters:
    - data: The input data object containing the subgraph information.
    - input_output_dict: A dictionary containing the input-output gene pairs and their indices.
    - input_gene: The input gene for which the perturbation is applied.
    - output_ids: A list of output gene indices of interest for the central spot.
    - perturb_level: The level of perturbation: 0-0th percentile, 1-25th percentile, 2-50th percentile, 3-75th percentile, 4-100th percentile.

    Returns:
    - data: The modified data object with the masked gene expression features and perturbation: mask (with masked central spot's gene expression features), and x (masked and perturbed gene expression features).
    """
    num_cells, num_gene_features = data.x.shape
    assert num_cells == 15, "The number of cells in the subgraph should be 15 (1 central spot + 14 neighbors)."
    assert num_gene_features == 14558 , "The number of gene features should be 14558 (the number of genes in the Visium dataset)."

    # Mask the central spot's gene expression features based on the input gene and all output genes of interest
    central_mask = torch.zeros(num_gene_features, dtype=torch.bool)
    central_mask[output_ids] = True  # Set the output gene indices to True in the mask


    # Apply the mask to the central spot's gene expression features
    masked_x = data.x.clone()
    masked_x[0, central_mask] = 0 # Mask the central spot's gene expression features

    # Generate perturbation values for the input gene
    if input_gene in input_output_dict.keys():
        min_value = input_output_dict[input_gene]['min_value']
        max_value = input_output_dict[input_gene]['max_value']
        perturbation_values = np.linspace(min_value, max_value, 5)[perturb_level] # Get the perturbation value based on the perturbation level
    else:
        raise ValueError(f"Input gene '{input_gene}' not found in the input-output dictionary.")

    # Set the neighboring gene expression levels for the input gene
    for i in range(1, num_cells):  # Start from 1 to skip the central spot
        if input_gene in input_output_dict.keys():
            input_index = input_output_dict[input_gene]['input_index']
            masked_x[i, input_index] = perturbation_values
        else:
            raise ValueError(f"Input gene '{input_gene}' not found in the input-output dictionary.")
        
    # Update the data object with the masked gene expression features
    data.x = masked_x
    return data

class DataGenerator(torch.utils.data.IterableDataset):
    """
    The DataGenerator class generates masked subgraphs on-the-fly for training the foundation model.
    """
    def __init__(self, data_dir, version, split, perturbation_gene, perturbation_dict, output_ids, perturbation_level, seed=9001):
        self.data_dir = data_dir # Directory containing the subgraph data
        self.version = version # Version of the dataset: V1 Small Intersection of small dataset, V2 UCE, V3 large intersection of small dataset, V4 HEST1k
        self.split = split # Split of the dataset: train, val, or test (80%, 10%, 10%, partition by each sample's subgraphs)
        self.perturbation_dict = perturbation_dict # CSV file containing the input-output gene pairs and their indices, min, and max values
        self.output_ids = output_ids # List of output gene indices of interest for the central spot
        self.files = [f for f in os.listdir(data_dir) if f.endswith(f"_{version}.pt")]
        self.perturbation_level = perturbation_level # Level of perturbation: 0-0th percentile, 1-25th percentile, 2-50th percentile, 3-75th percentile, 4-100th percentile
        self.perturbation_gene = perturbation_gene # The input gene for perturbation, which is used to set the neighboring spot's gene expression levels
        random.seed(seed)
        random.shuffle(self.files)
        total = len(self.files)
        if split == 'train':
            self.files = self.files[:int(0.8 * total)]
        elif split == 'val':
            self.files = self.files[int(0.8 * total):int(0.9 * total)]
            np.save('HEST1k_Validation Files.npy', np.array(self.files))
        elif split == 'test':
            self.files = self.files[int(0.9 * total):]
            np.save('HEST1k_Test Files.npy', np.array(self.files))

    def __iter__(self):
        for file in self.files:
            # weights_only=False is needed for PyTorch Geometric data objects (PyTorch 2.6+ default changed)
            subgraphs = torch.load(os.path.join(self.data_dir, file), weights_only=False) # A list of subgraphs (Data type): (number of spots), each subgraph has (data.x, data.edge, shape is (15, 14558))
            labels = [file.split('_')[0]] * len(subgraphs) # Number of central spots
            for subgraph, label in zip(subgraphs, labels):
                subgraph.y = label # One label per subgraph for the central spot
                subgraph = insilico_perturbation(data = subgraph, input_output_dict = self.perturbation_dict, input_gene = self.perturbation_gene, output_ids = self.output_ids, perturb_level = self.perturbation_level) # Apply the perturbation to the subgraph
                yield subgraph


# Function to load a pre-trained model
def load_model(checkpoint_path, version):
    # Load the checkpoint
    # weights_only=False is needed for PyTorch Geometric models and custom objects
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Extract the saved parameters
    hidden_channels = checkpoint['hidden_channels']
    learning_rate = checkpoint['learning_rate']
    batch_size = checkpoint['batch_size']
    
    # Define the model structure
    try:
        in_channels = checkpoint['in_channels']  # Update this if needed
    except KeyError:
        if version == 'v2':
            in_channels = 1280  # Default value for v2 models
        elif version == 'v3':
            in_channels = 11022  # Default value for v3 models
        elif version == 'v4':
            in_channels = 14558  # Default value for v4 models
        else:
            in_channels = 11022
    try:    
        num_gene_features = checkpoint['num_gene_features']  # Update this if needed
    except KeyError:
        if version == 'v2':
            num_gene_features = 1280
        elif version == 'v3':
            num_gene_features = 11022 # Default value for v2 models
        elif version == 'v4':
            num_gene_features = 14558  # Default value for v4 models
        else:
            num_gene_features = 11022

    model = PretrainModel(in_channels, hidden_channels, num_gene_features)
    
    # Load the state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint

def evaluate_perturbation_with_generator(
    model_name, 
    test_version, 
    device=None,
    perturbation_levels=None,
    output_dir_test=None,
    output_dir_val=None,
    csv_file_path=None,
    baseline_seed=42):
    """
    Evaluate perturbation with randomly selected baseline genes instead of downstream genes.
    
    Parameters:
    - baseline_seed: Random seed for selecting baseline genes (for reproducibility)
    """

    # 1. Define the specifications for the pre-trained model
    print('Loading model...', model_name)
    model_dir  = "/scratch/users/xzhan96/HEST1k_selected_models"
    model_path = os.path.join(model_dir, model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_version = str(model_name.split("_")[0])
    test_version = str(test_version)
    mask_percentage = float(model_name.split("mask_")[1].split('_')[0])
    batch_size = int(model_name.split("batch_")[1].split('.pt')[0])
    
    # Determine total number of genes based on version
    if test_version == 'v4':
        total_num_genes = 14558
    elif test_version == 'v3':
        total_num_genes = 11022
    elif test_version == 'v2':
        total_num_genes = 1280
    else:
        total_num_genes = 14558  # Default to v4
    
    # Set default CSV file path if not specified
    if csv_file_path is None:
        csv_file_path = '/scratch/users/xzhan96/Paired Genes and Statistics.csv'
    
    print(f"Loading CSV file from: {csv_file_path}")
    perturbation_dict, output_ids = load_csv(csv_file_path) # Load the input-output gene pairs and their indices, min, and max values

    # 2. Load the model and define the data generators with in-silico perturbations
    model, optimizer, scheduler, checkpoint = load_model(model_path, version=model_version)

    data_dir_dict = {
        'v1': "/scratch/users/xzhan96/Visium/gnn_subgraphs/v1_intersection",
        'v2': "/scratch/users/xzhan96/Visium/gnn_subgraphs/v2_UCE",
        'v3': "/scratch/users/xzhan96/Visium/gnn_subgraphs/v3_large_intersection",
        'v4': "/scratch/users/xzhan96/Visium/gnn_subgraphs/v4_HEST1k",
        'v5': "/scratch/users/xzhan96/Visium/gnn_subgraphs/v5_hest1k_holdouttest",
    }
    data_dir = data_dir_dict[test_version]

    # Set default perturbation levels if not specified
    if perturbation_levels is None:
        perturbation_levels = [0, 1, 2, 3, 4]  # Default: all levels (0th, 25th, 50th, 75th, 100th percentiles)
    
    # Set default output directories if not specified
    if output_dir_test is None:
        output_dir_test = '/scratch/users/xzhan96/HEST1k_test_perturbation_baseline'
    if output_dir_val is None:
        output_dir_val = '/scratch/users/xzhan96/HEST1k_val_perturbation_baseline'
    
    # Create output directories
    os.makedirs(output_dir_test, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)
    
    print(f"Running perturbation levels: {perturbation_levels}")
    print(f"Test output directory: {output_dir_test}")
    print(f"Validation output directory: {output_dir_val}")
    print(f"BASELINE MODE: Using randomly selected genes instead of downstream genes")
    print(f"Baseline selection seed: {baseline_seed}")

    # Create data generators with peturbations and masks, calculate the correlation between the input gene and the output gene expression levels
    for perturbation_gene in perturbation_dict.keys():
        print(f"Perturbation Gene: {perturbation_gene}")

        # Get downstream genes info for this input gene
        output_gene_indices = perturbation_dict[perturbation_gene]['output_indices']
        num_downstream_genes = len(output_gene_indices)
        input_index = perturbation_dict[perturbation_gene]['input_index']
        
        # Select random baseline genes (same number as downstream genes)
        # Exclude: input gene index and downstream gene indices
        exclude_indices = np.concatenate([[input_index], output_gene_indices])
        
        # Use a unique seed per input gene for reproducibility but different baselines
        gene_specific_seed = baseline_seed + hash(perturbation_gene) % 10000
        baseline_indices = select_random_baseline_genes(
            num_genes_to_select=num_downstream_genes,
            exclude_indices=exclude_indices,
            total_num_genes=total_num_genes,
            seed=gene_specific_seed
        )
        
        print(f"Downstream genes for {perturbation_gene}: {num_downstream_genes} genes")
        print(f"Baseline genes selected: {len(baseline_indices)} genes (random selection)")
        print(f"Baseline gene indices: {baseline_indices[:10]}..." if len(baseline_indices) > 10 else f"Baseline gene indices: {baseline_indices}")

        for perturbation_level in perturbation_levels:  # Only run specified perturbation levels
            print(f"Perturbation Level: {perturbation_level}")
            perturbation_value = np.linspace(
                perturbation_dict[perturbation_gene]['min_value'], 
                perturbation_dict[perturbation_gene]['max_value'], 
                5)[perturbation_level]
            print(f"Perturbated Gene Expression Value: {perturbation_value}")

            # 3. Create the data generators for validation, and testing
            # Use baseline gene indices instead of downstream gene indices
            val_dataset = DataGenerator(data_dir, test_version, 'val', perturbation_gene, perturbation_dict, baseline_indices, perturbation_level)
            test_dataset = DataGenerator(data_dir, test_version, 'test', perturbation_gene, perturbation_dict, baseline_indices, perturbation_level)

            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # 4. Make the prediction under the perturbation and store the result on the test set
            print(f"Evaluating model: {model_name} on {test_version} with perturbation gene: {perturbation_gene} at level {perturbation_level} (BASELINE)")
            log_memory_usage("Before Evaluation")

            model = model.to(device)
            model.eval()
            y_test_baseline_gene_expressions = [] # Baseline gene expressions for the central spot

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    predictions, _ = model(data)
                    # Collect predictions for baseline genes
                    y_test_baseline_gene_expressions.append(predictions.cpu()[::15, baseline_indices]) # Shape: [batch_size, num_baseline_genes]
            
            y_test_baseline_gene_expressions = torch.cat(y_test_baseline_gene_expressions, dim=0)  # Shape: [num_samples, num_baseline_genes]

            # 5. Store the baseline gene expressions with input gene name, perturbation level in the file name.
            np.save(f"{output_dir_test}/version_{test_version}_{mask_percentage}_baseline_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_test_baseline_gene_expressions.numpy())
            log_memory_usage("After Evaluation")

            # 6. Free up memory
            del test_dataset, test_loader, y_test_baseline_gene_expressions

            # 7. Evaluate the model performance on the validation set
            print(f"Evaluating model: {model_name} on {test_version} with perturbation gene: {perturbation_gene} at level {perturbation_level} on validation set (BASELINE)")
            log_memory_usage("Before Validation Evaluation")
            y_val_baseline_gene_expressions = []

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    predictions, _ = model(data)
                    # Collect predictions for baseline genes
                    y_val_baseline_gene_expressions.append(predictions.cpu()[::15, baseline_indices])  # Shape: [batch_size, num_baseline_genes]
            y_val_baseline_gene_expressions = torch.cat(y_val_baseline_gene_expressions, dim=0)  # Shape: [num_samples, num_baseline_genes]

            # Store the baseline gene expressions with input gene name, perturbation level in the file name.
            np.save(f"{output_dir_val}/version_{test_version}_{mask_percentage}_baseline_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_val_baseline_gene_expressions.numpy())
            del val_dataset, val_loader, y_val_baseline_gene_expressions

            log_memory_usage("After Validation Evaluation")
            

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a foundation model for predicting masked gene expression features using BASELINE (random) genes.")
    parser.add_argument('--model_name', type=str, default='v4_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt', help='Name of Pre-trained Model')
    parser.add_argument('--test_version', type=str, default='v4', help='Version of the test data')
    parser.add_argument('--perturbation_levels', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Perturbation levels to run (0=0th, 1=25th, 2=50th, 3=75th, 4=100th percentile). Default: all levels.')
    parser.add_argument('--output_dir_test', type=str, default=None, help='Output directory for test set results. Default: /scratch/users/xzhan96/HEST1k_test_perturbation_baseline')
    parser.add_argument('--output_dir_val', type=str, default=None, help='Output directory for validation set results. Default: /scratch/users/xzhan96/HEST1k_val_perturbation_baseline')
    parser.add_argument('--csv_file', type=str, default=None, help='Path to CSV file with paired genes. Default: /scratch/users/xzhan96/Paired Genes and Statistics.csv')
    parser.add_argument('--baseline_seed', type=int, default=42, help='Random seed for selecting baseline genes. Default: 42')
    args = parser.parse_args()

    # Run the evaluation pipeline
    evaluate_perturbation_with_generator(
        model_name=args.model_name,
        test_version=args.test_version,
        device="cuda" if torch.cuda.is_available() else "cpu",
        perturbation_levels=args.perturbation_levels,
        output_dir_test=args.output_dir_test,
        output_dir_val=args.output_dir_val,
        csv_file_path=args.csv_file,
        baseline_seed=args.baseline_seed
    )

if __name__ == "__main__":
    main()

