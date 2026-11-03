# =====================================================================
# EdgeSecAI: Core Training & Weighted FedAvg Engine (federated_training.py)
# =====================================================================

import copy
import gc
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fresh_model():
    """Loads and fixes a fresh DistilBERT model for Opacus compatibility."""
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model

def configure_layers(model, use_dp=False):
    """
    Freezes lower layers to prevent 66M-parameter DP noise explosion and OOM.
    Trains top transformer layer (layer.4, layer.5) + classification head.
    """
    for name, param in model.named_parameters():
        if use_dp:
            if any(k in name for k in ["transformer.layer.5", "pre_classifier", "classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            if any(k in name for k in ["transformer.layer.4", "transformer.layer.5", "pre_classifier", "classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model

def compute_class_weights(dataset, indices, device):
    """Computes inverse class weights to prevent 58.16% majority class collapse."""
    labels = [int(dataset[i]['labels']) for i in indices]
    counts = np.bincount(labels, minlength=2)
    total = max(len(labels), 1)
    weights = [total / (2.0 * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float, device=device)

def weighted_fedavg(client_states, client_sizes):
    """Properly aggregates client state_dicts weighted by client dataset sizes."""
    total_size = sum(client_sizes)
    avg_state = OrderedDict()
    first_state = client_states[0]

    for key in first_state.keys():
        if first_state[key].is_floating_point():
            avg_state[key] = sum(
                state[key].cpu().float() * (size / total_size)
                for state, size in zip(client_states, client_sizes)
            )
        else:
            # Keep integer buffers (like position_ids) intact
            avg_state[key] = first_state[key].cpu().clone()
    return avg_state

def train_client_edgesecai(
    global_state_dict, dataset, indices,
    epochs=2, batch_size=32, use_dp=False, epsilon=1.0, delta=1e-5, mu=0.0
):
    """Trains a single client with optional DP-SGD and FedProx."""
    if len(indices) == 0:
        return copy.deepcopy(global_state_dict), 0.0

    # Initialize client model with global state
    model = get_fresh_model()
    model.load_state_dict(global_state_dict, strict=True)
    model = configure_layers(model, use_dp=use_dp)
    model.to(device)
    model.train()

    # Balanced Loss Function
    class_weights = compute_class_weights(dataset, indices, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # DataLoader
    client_subset = Subset(dataset, indices)
    eff_bs = min(batch_size, len(client_subset))
    dataloader = DataLoader(
        client_subset,
        batch_size=eff_bs,
        shuffle=True,
        drop_last=use_dp
    )

    # Optimizer (Higher LR for frozen-layer DP to overcome clipping)
    lr = 5e-4 if use_dp else 1e-4
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Save global snapshot for FedProx
    global_trainable = {
        name: param.detach().clone()
        for name, param in model.named_parameters() if param.requires_grad
    }

    privacy_engine = None
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=1.0,
        )

    # Training Loop
    total_loss, steps = 0.0, 0
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            # FedProx Proximal Regularization Term
            if mu > 0.0:
                proximal_term = 0.0
                base_model = model._module if use_dp else model
                for name, w in base_model.named_parameters():
                    if w.requires_grad and name in global_trainable:
                        proximal_term += torch.sum((w - global_trainable[name]) ** 2)
                loss = loss + (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    spent_eps = privacy_engine.get_epsilon(delta) if use_dp else 0.0
    avg_loss = total_loss / max(steps, 1)

    final_state = copy.deepcopy(
        model._module.state_dict() if use_dp else model.state_dict()
    )
    final_state = OrderedDict({k: v.cpu() for k, v in final_state.items()})

    print(f"      Loss: {avg_loss:.4f} | Samples: {len(indices)} | DP: {use_dp} (Eps: {spent_eps:.3f}) | FedProx mu: {mu}")

    del model, optimizer, dataloader, privacy_engine, global_trainable
    torch.cuda.empty_cache()
    gc.collect()

    return final_state, spent_eps