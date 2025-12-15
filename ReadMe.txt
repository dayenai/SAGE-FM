SAGE-FM: A lightweight and interpretable spatial transcriptomics foundation model
Authors: Xianghao Zhan, Jingyu Xu
Contact: xzhan96@stanford.edu

This repository is associated with the manuscript entitled “SAGE-FM: A lightweight and interpretable spatial transcriptomics foundation model.”

Model Pre-training
To ensure reproducibility, the code used to pre-train the graph convolutional neural network (GCN) on the masked central-spot gene expression prediction task using the HEST1k dataset is provided in the Pre-training/ subfolder:
•	SAGE-FM_Training_HEST1k.py

We also provide:
•	the pre-trained model checkpoint
HEST1k_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt
•	the 14,558-gene list in the Pre-trained Model/ subfolder.

Evaluation and Benchmarking
The code used for evaluation tasks is provided in the Evaluation and Benchmarking/ subfolder, including:
•	SAGE-FM_Pretraining Evaluation and Get Embeddings_HEST1k.py (evaluate masked gene prediction and export embeddings)
•	SAGE-FM_Missing Gene Imputation.py (systematic missingness imputation)
•	SAGE-FM_In silico Perturbation.py (in silico perturbation experiments)

Run Inference on New Data
For users who would like to infer SAGE-FM embeddings for a new Visium spatial transcriptomics dataset, please use the Post-training Inference/ folder.
Step 1: Generate subgraphs
First, run Subgraph Generation.ipynb to preprocess your dataset and generate 15-spot neighborhood subgraphs.
Step 2: Run inference to obtain embeddings
After generating subgraphs, run SAGE-FM_New Dataset Inference.py to produce SAGE-FM embeddings for each subgraph.

Important notes (based on the current script implementation):
•	The script expects your subgraph .pt files to be located in a dataset-specific directory (e.g., ./gnn_subgraphs/GBM) and the filenames should end with _<DATASET>.pt (e.g., sample1_GBM.pt). 
•	The script currently looks for the model checkpoint under ./HEST1k_selected_models/. 
•	The embeddings (and labels) are saved to ./HEST1k_holdouttest_subgraph_embeddings/ as .npy files. 

Example command
1) Put your generated subgraph .pt files here:
./gnn_subgraphs/GBM/
 (each file should end with _GBM.pt, e.g., sampleA_GBM.pt)

2) Put the pretrained checkpoint here:
./HEST1k_selected_models/HEST1k_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt

python "Post-training Inference/SAGE-FM_New Dataset Inference.py" \
  --model_name "HEST1k_epoch_126_hidden_1024_512_512_512_1024_lr_5e-05_wd_0.0001_mask_0.3_batch_64.pt" \
  --test_dataset "GBM"
Outputs: the script will write (at minimum) the embedding file and label file to:
•	./HEST1k_holdouttest_subgraph_embeddings/<model_name>_subgraph_embedding.npy
•	./HEST1k_holdouttest_subgraph_embeddings/<model_name>_subgraph_labels.npy 
If your dataset name is not GBM, you can either (a) rename your dataset key to GBM, or (b) edit the data_dir_dict inside SAGE-FM_New Dataset Inference.py to add your dataset name and path. 

Example Data
To help users understand the expected input and output formats, we include one sample from the OSCC downstream task in the paper in the Example Data/ subfolder.

