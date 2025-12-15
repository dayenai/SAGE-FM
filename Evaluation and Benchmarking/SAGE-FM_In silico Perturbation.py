"""
Xianghao Zhan, 12/14/2025
This code use a pre-trained spatial foundation model for in silico perturbation study using the test and validation sets.
This code will load a csv file with paired genes (input-output gene pairs). The csv files have columns including the gene names, the indices of the columns and the min/max values of the input gene expression levels across the entire dataset.
Then, it will artificially upregulate/downregulate the neighboring spots' input gene expression levels and record the central spot's output gene expression levels.
Perturbation will be setting the input gene expression levels to be 5 values between the min and max values of the input gene expression levels.
Paired input-output genes are expected to exhibit positive correlation while unpaired input-output gene should exhibit no correlations.
This code is designed to evaluate whether the model learned gene regulatory networks.
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
import pandas as pd
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
    """
    df = pd.read_csv(file_path)
    input_output_dict = {}
    output_ids = np.array(df["Output ID"].tolist())  # Numpy array of output indices
    for _, row in df.iterrows():
        input_gene = row['Input Gene']
        output_gene = row['Output Gene']
        input_index = int(row['Input ID'])
        output_index = int(row['Output ID'])
        min_value = float(row['0th'])
        max_value = float(row['100th'])
        
        # Store the information in a dictionary
        input_output_dict[input_gene] = {
            'output_gene': output_gene,
            'unmatched_gene': [g for g in df["Output Gene"].tolist() if g != output_gene],
            'input_index': input_index,
            'output_index': output_index,
            'unmatched_index': [i for i in df["Output ID"].tolist() if i != output_index],
            'min_value': min_value,
            'max_value': max_value
        }
    return input_output_dict, output_ids


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
    def __init__(self, data_dir, dataset, split, perturbation_gene, perturbation_dict, output_ids, perturbation_level, seed=9001):
        self.data_dir = data_dir # Directory containing the subgraph data
        self.dataset = dataset # Name of the dataset: HEST1k
        self.split = split # Split of the dataset: train, val, or test (80%, 10%, 10%, partition by each sample's subgraphs)
        self.perturbation_dict = perturbation_dict # CSV file containing the input-output gene pairs and their indices, min, and max values
        self.output_ids = output_ids # List of output gene indices of interest for the central spot
        self.files = [f for f in os.listdir(data_dir) if f.endswith(f"_{dataset}.pt")]
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
            subgraphs = torch.load(os.path.join(self.data_dir, file)) # A list of subgraphs (Data type): (number of spots), each subgraph has (data.x, data.edge, shape is (15, 14558))
            labels = [file.split('_')[0]] * len(subgraphs) # Number of central spots
            for subgraph, label in zip(subgraphs, labels):
                subgraph.y = label # One label per subgraph for the central spot
                subgraph = insilico_perturbation(data = subgraph, input_output_dict = self.perturbation_dict, input_gene = self.perturbation_gene, output_ids = self.output_ids, perturb_level = self.perturbation_level) # Apply the perturbation to the subgraph
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
        if dataset == 'HEST1k':
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

def evaluate_perturbation_with_generator(
    model_name, 
    test_dataset, 
    device=None):

    # 1. Define the specifications for the pre-trained model
    print('Loading model...', model_name)
    model_dir  = "./Model"
    model_path = os.path.join(model_dir, model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dataset = str(model_name.split("_")[0])
    test_dataset = str(test_dataset)
    mask_percentage = float(model_name.split("mask_")[1].split('_')[0])
    batch_size = int(model_name.split("batch_")[1].split('.pt')[0])
    perturbation_dict, output_ids = load_csv('./Paired Genes and Statistics.csv') # Load the input-output gene pairs and their indices, min, and max values

    # 2. Load the model and define the data generators with in-silico perturbations
    model, optimizer, scheduler, checkpoint = load_model(model_path, dataset=model_dataset)

    data_dir_dict = {
        'HEST1k': "./gnn_subgraphs/HEST1k",
    }
    data_dir = data_dir_dict[test_dataset]

    # Create data generators with peturbations and masks, calculate the correlation between the input gene and the output gene expression levels
    for perturbation_gene in perturbation_dict.keys():
        print(f"Perturbation Gene: {perturbation_gene}")

        for perturbation_level in range(5):  # 0-4 for 0th, 25th, 50th, 75th, and 100th percentiles
            print(f"Perturbation Level: {perturbation_level}")
            perturbation_value = np.linspace(
                perturbation_dict[perturbation_gene]['min_value'], 
                perturbation_dict[perturbation_gene]['max_value'], 
                5)[perturbation_level]
            print(f"Perturbated Gene Expression Value: {perturbation_value}")
            
            # Get the output gene index and unmatched gene indices
            output_gene_id = perturbation_dict[perturbation_gene]['output_index'] # One int value
            unmatched_gene_id = np.array(perturbation_dict[perturbation_gene]['unmatched_index'], dtype=int) # Int array, the indices of the unmatched output genes

            # 3. Create the data generators for validation, and testing
            val_dataset = DataGenerator(data_dir, test_dataset, 'val', perturbation_gene, perturbation_dict, output_ids, perturbation_level)
            test_dataset = DataGenerator(data_dir, test_dataset, 'test', perturbation_gene, perturbation_dict, output_ids, perturbation_level)

            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # 4. Make the prediction under the perturbation and store the result on the test set
            print(f"Evaluating model: {model_name} on {test_dataset} with perturbation gene: {perturbation_gene} at level {perturbation_level}")
            log_memory_usage("Before Evaluation")

            model = model.to(device)
            model.eval()
            y_test_matched_gene_expressions = [] # Matched gene expressions for the central spot w.r.t. the input gene
            y_test_unmatched_gene_expressions = [] # Unmatched gene expressions for the central spot w.r.t. the input gene

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    predictions, _ = model(data)
                    y_test_matched_gene_expressions.append(predictions.cpu()[::15, output_gene_id]) # Matched gene expressions for the central spot w.r.t. the input gene (only central spots)
                    y_test_unmatched_gene_expressions.append(predictions.cpu()[::15, unmatched_gene_id]) # Unmatched gene expressions for the central spot w.r.t. the input gene (only central spots)
            
            y_test_matched_gene_expressions = torch.cat(y_test_matched_gene_expressions, dim=0)
            y_test_unmatched_gene_expressions = torch.cat(y_test_unmatched_gene_expressions, dim=0)

            # 5. Store the matched and unmatched gene expressions with input gene name, perturbation level in the file name.
            os.makedirs('./HEST1k_test_perturbation', exist_ok=True)
            np.save(f"./HEST1k_test_perturbation/dataset_{test_dataset}_{mask_percentage}_matched_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_test_matched_gene_expressions.numpy())
            np.save(f"./HEST1k_test_perturbation/dataset_{test_dataset}_{mask_percentage}_unmatched_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_test_unmatched_gene_expressions.numpy())
            log_memory_usage("After Evaluation")

            # 6. Free up memory
            del test_dataset, test_loader, y_test_matched_gene_expressions, y_test_unmatched_gene_expressions

            # 7. Evaluate the model performance on the validation set
            print(f"Evaluating model: {model_name} on {test_dataset} with perturbation gene: {perturbation_gene} at level {perturbation_level} on validation set")
            log_memory_usage("Before Validation Evaluation")
            y_val_matched_gene_expressions = []
            y_val_unmatched_gene_expressions = [] 

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    predictions, _ = model(data)
                    y_val_matched_gene_expressions.append(predictions.cpu()[::15, output_gene_id])
                    y_val_unmatched_gene_expressions.append(predictions.cpu()[::15, unmatched_gene_id])
            y_val_matched_gene_expressions = torch.cat(y_val_matched_gene_expressions, dim=0)
            y_val_unmatched_gene_expressions = torch.cat(y_val_unmatched_gene_expressions, dim=0)

            # Store the matched and unmatched gene expressions with input gene name, perturbation level in the file name.
            os.makedirs('./HEST1k_val_perturbation', exist_ok=True)
            np.save(f"./HEST1k_val_perturbation/dataset_{test_dataset}_{mask_percentage}_matched_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_val_matched_gene_expressions.numpy())
            np.save(f"./HEST1k_val_perturbation/dataset_{test_dataset}_{mask_percentage}_unmatched_gene_expressions_{perturbation_gene}_level_{perturbation_level}_perturbation_value_{perturbation_value}.npy", y_val_unmatched_gene_expressions.numpy())
            del val_dataset, val_loader, y_val_matched_gene_expressions, y_val_unmatched_gene_expressions

            log_memory_usage("After Validation Evaluation")
            

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a foundation model for predicting masked gene expression features.")
    parser.add_argument('--model_name', type=str, default='HEST1k_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt', help='Name of Pre-trained Model')
    parser.add_argument('--test_dataset', type=str, default='HEST1k', help='Name of the test data')
    args = parser.parse_args()

    # Run the evaluation pipeline
    evaluate_perturbation_with_generator(
        model_name=args.model_name,
        test_dataset=args.test_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

if __name__ == "__main__":
    main()
