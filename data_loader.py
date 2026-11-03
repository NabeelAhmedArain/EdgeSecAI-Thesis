# =====================================================================
# EdgeSecAI: Dataset Loading, Preprocessing & Formatting (data_loader.py)
# =====================================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import Counter

class TextDataset(Dataset):
    """Custom PyTorch Dataset for Text Classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_and_prepare_data(data_dir='./data'):
    """
    Loads OLID and HatEval datasets, combines them, and tokenizes.
    Returns the tokenized encodings, labels, tokenizer, and a PyTorch Dataset object.
    """
    print("--- 1. Loading Datasets ---")
    
    olid_path = f'{data_dir}/offenseval-training-v1.tsv'
    hateval_train_path = f'{data_dir}/hateval_train.csv'
    hateval_dev_path = f'{data_dir}/hateval_dev.csv'

    # Load OLID Dataset
    olid_df = pd.read_csv(olid_path, sep='\t')
    print(f"OLID Loaded: {olid_df.shape[0]} rows")
    olid_texts = olid_df['tweet'].tolist()
    olid_labels = [1 if val == 'OFF' else 0 for val in olid_df['subtask_a'].tolist()]

    # Load HatEval Train & Dev Datasets
    hateval_train = pd.read_csv(hateval_train_path)
    hateval_dev = pd.read_csv(hateval_dev_path)
    hateval_df = pd.concat([hateval_train, hateval_dev], ignore_index=True)
    print(f"HatEval Loaded (Train + Dev): {hateval_df.shape[0]} rows")

    hateval_texts = hateval_df['text'].tolist()
    hateval_labels = hateval_df['HS'].tolist()

    # Combine Datasets
    print("\n--- 2. Combining Datasets into a Unified Corpus ---")
    all_texts = olid_texts + hateval_texts
    all_labels = olid_labels + hateval_labels
    print(f"Total Combined Corpus Size: {len(all_texts)} samples")

    # Initialize Tokenizer
    print("\n--- 3. Initializing DistilBERT Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("Tokenizing texts (max_length=128)...")
    encodings = tokenizer(all_texts, truncation=True, padding=True, max_length=128)
    print("Tokenization complete!")
    print("Class Distribution:", Counter(all_labels))
    
    # Initialize PyTorch Dataset
    print("\n--- 4. Initializing PyTorch Dataset Object ---")
    dataset = TextDataset(encodings, all_labels)
    print(f"Dataset object successfully initialized! Total samples: {len(dataset)}")
    
    return encodings, all_labels, tokenizer, dataset

if __name__ == "__main__":
    # Test execution
    encodings, labels, tokenizer, dataset = load_and_prepare_data()

def create_dirichlet_split(dataset, num_clients=5, alpha=0.5):
    """Creates a Non-IID Dirichlet split for Federated Learning clients."""
    import numpy as np
    print(f"Generating Non-IID Dirichlet (alpha={alpha}) split for {num_clients} clients...")
    np.random.seed(42)
    all_labels = np.array([int(dataset[i]['labels']) for i in range(len(dataset))])
    
    client_idcs = [[] for _ in range(num_clients)]
    for k in range(2): # Binary classification
        idx_k = np.where(all_labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        for cid, idx_split in enumerate(np.split(idx_k, splits)):
            client_idcs[cid].extend(idx_split.tolist())
            
    clients_map = {i: client_idcs[i] for i in range(num_clients)}
    print(f"✅ Client Splits Generated -> Sizes: {[len(v) for v in clients_map.values()]}")
    return clients_map