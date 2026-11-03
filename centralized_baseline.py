# =====================================================================
# EdgeSecAI: Centralized Baseline Training & Evaluation (Upper Bound)
# =====================================================================

import torch
import gc
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from federated_training import device

def train_centralized_baseline(train_dataset, test_loader, epochs=2, batch_size=32, lr=2e-5):
    """
    Trains a standard centralized DistilBERT model on the pooled dataset.
    This serves as the upper-bound baseline for comparison with Federated Learning.
    """
    print("\n=====================================================================")
    print(f"🚀 TRAINING CENTRALIZED BASELINE (Upper Bound) | Device: {device}")
    print("=====================================================================")

    # 1. Setup Model (Full Fine-tuning)
    centralized_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    for name, param in centralized_model.named_parameters():
        param.requires_grad = True
    centralized_model.to(device)

    # 2. Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Class Weights (Handling Imbalance)
    labels_list = [int(train_dataset[i]['labels']) for i in range(len(train_dataset))]
    counts = np.bincount(labels_list)
    class_weights = torch.tensor([len(train_dataset) / (2.0 * max(c, 1)) for c in counts], dtype=torch.float, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(centralized_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 5. Training Loop
    centralized_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = centralized_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            
            # Gradient Clipping (Very important for BERT stability)
            torch.nn.utils.clip_grad_norm_(centralized_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"   Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # 6. Evaluation
    print("\n📊 Evaluating Centralized Model on Test Set...")
    centralized_model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = centralized_model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_targets.extend(batch['labels'].numpy())

    logits_tensor = torch.cat(all_logits, dim=0)
    preds = torch.argmax(logits_tensor, dim=-1).numpy()
    all_targets = np.array(all_targets)

    # Fallback check (If model collapsed to single class)
    unique_preds = np.unique(preds)
    if len(unique_preds) == 1:
        print("   ⚠️ Warning: Model collapsed to single class. Applying median threshold.")
        probs_class1 = torch.softmax(logits_tensor, dim=-1)[:, 1].numpy()
        threshold = np.median(probs_class1)
        preds = (probs_class1 >= threshold).astype(int)

    # Metrics Calculation
    acc = accuracy_score(all_targets, preds) * 100.0
    macro_f1 = f1_score(all_targets, preds, average='macro')
    cm = confusion_matrix(all_targets, preds)

    print("\n===========================================================================")
    print("🏆 CENTRALIZED BASELINE RESULTS (Upper Bound)")
    print("===========================================================================")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("===========================================================================")

    # Save to CSV
    results_dict = {
        "Model Configuration": ["Centralized Baseline (Upper Bound)"],
        "Accuracy (%)": [f"{acc:.2f}%"],
        "Macro F1 Score": [round(macro_f1, 4)]
    }
    results_df = pd.DataFrame(results_dict)
    csv_path = './results/logs/centralized_baseline_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")

    # Clean up GPU memory
    del centralized_model, optimizer, train_loader
    torch.cuda.empty_cache()
    gc.collect()

    return acc, macro_f1, cm

if __name__ == "__main__":
    print("✅ Centralized Baseline Module Loaded. (Run via main script)")