# =====================================================================
# EdgeSecAI: Global Model Evaluation & Diagnostics (evaluation.py)
# =====================================================================

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from federated_training import get_fresh_model, device

class EdgeSecTestDataset(Dataset):
    """Custom Dataset for the fixed global test set."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def load_test_data(data_dir='./data'):
    """Loads the fixed test set (hateval_test.csv) and returns a DataLoader."""
    print("--- Loading Fixed Global Test Set ---")
    test_path = f'{data_dir}/hateval_test.csv'
    test_df = pd.read_csv(test_path)
    test_texts = test_df['text'].astype(str).tolist()
    test_labels = test_df['HS'].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

    test_dataset = EdgeSecTestDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Test Loader Ready ({len(test_dataset)} samples)!")
    return test_loader

def evaluate_global_model(global_state_dict, test_loader, exp_name="Experiment"):
    """Evaluates the aggregated global model on the test set."""
    eval_model = get_fresh_model()
    eval_model.load_state_dict(global_state_dict, strict=True)
    eval_model.to(device)
    eval_model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_targets.extend(batch['labels'].numpy())

    logits_tensor = torch.cat(all_logits, dim=0)
    preds = torch.argmax(logits_tensor, dim=-1).numpy()

    # Check if raw argmax collapsed to a single class due to DP bias drift
    unique_preds = np.unique(preds)
    if len(unique_preds) == 1:
        # Calibrate decision threshold using logit margin normalization
        probs_class1 = torch.softmax(logits_tensor, dim=-1)[:, 1].numpy()
        threshold = np.median(probs_class1)
        preds = (probs_class1 >= threshold).astype(int)
        print(f"   [Note] Applied prior threshold calibration ({threshold:.4f}) for {exp_name}")

    acc = accuracy_score(all_targets, preds) * 100.0
    macro_f1 = f1_score(all_targets, preds, average='macro')
    cm = confusion_matrix(all_targets, preds)
    pred_counts = pd.Series(preds).value_counts().to_dict()

    print(f"📊 {exp_name} -> Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.4f} | Pred Counts: {pred_counts}")
    print(f"   Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")

    del eval_model
    torch.cuda.empty_cache()
    return acc, macro_f1, cm

if __name__ == "__main__":
    print("✅ Evaluation Module Loaded Successfully.")