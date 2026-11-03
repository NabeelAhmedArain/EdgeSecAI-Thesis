# =====================================================================
# EdgeSecAI: Main Ablation Study & Execution Script (run_experiments.py)
# =====================================================================

import os
import copy
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from data_loader import load_and_prepare_data
from federated_training import train_client_edgesecai, weighted_fedavg, get_fresh_model
from evaluation import load_test_data, evaluate_global_model

def create_dirichlet_split(dataset, num_clients=5, alpha=0.5):
    """Creates a Non-IID Dirichlet split for Federated Learning clients."""
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

def main():
    # 1. Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 3           # Change to 10-15 for final full run
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    TARGET_EPSILON = 1.0

    # Ensure results directory exists
    os.makedirs('./results/logs', exist_ok=True)

    # 2. Load Data
    print("\n--- Initializing Data ---")
    _, _, _, train_dataset = load_and_prepare_data(data_dir='./data')
    test_loader = load_test_data(data_dir='./data')
    
    # 3. Create Client Splits
    clients_map = create_dirichlet_split(train_dataset, num_clients=NUM_CLIENTS, alpha=0.5)

    # 4. Experiment Configuration
    experiments_config = [
        ("E1: Baseline (FedAvg)",   False, 0.0,  0.0),
        ("E2: FedProx Only",        False, 0.01, 0.0),
        ("E3: DP Only",             True,  0.0,  TARGET_EPSILON),
        ("E4: FedProx + DP (Main)", True,  0.01, TARGET_EPSILON),
    ]

    saved_global_weights = {}
    summary_rows = []

    # Set global seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    initial_base_state = OrderedDict({k: v.cpu() for k, v in get_fresh_model().state_dict().items()})

    # 5. Main Training Loop
    for exp_title, use_dp_flag, mu_val, eps_val in experiments_config:
        print(f"\n=====================================================================")
        print(f"🚀 STARTING {exp_title} | DP={use_dp_flag} | mu={mu_val}")
        print(f"=====================================================================")

        global_state = copy.deepcopy(initial_base_state)
        max_eps_spent = 0.0

        for r in range(1, NUM_ROUNDS + 1):
            print(f"\n--- Round {r}/{NUM_ROUNDS} ---")
            round_states = []
            round_sizes = []

            for cid in list(clients_map.keys()):
                c_indices = clients_map[cid]
                print(f"   [Client {cid}] Training...")
                c_state, c_eps = train_client_edgesecai(
                    global_state_dict=global_state,
                    dataset=train_dataset,
                    indices=c_indices,
                    epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    use_dp=use_dp_flag,
                    epsilon=eps_val,
                    delta=1e-5,
                    mu=mu_val
                )
                round_states.append(c_state)
                round_sizes.append(len(c_indices))
                max_eps_spent = max(max_eps_spent, c_eps)

            global_state = weighted_fedavg(round_states, round_sizes)

        # Evaluate and Save
        acc, macro_f1, cm = evaluate_global_model(global_state, test_loader, exp_name=exp_title)
        saved_global_weights[exp_title] = copy.deepcopy(global_state)

        summary_rows.append({
            "Experiment": exp_title,
            "FedProx (mu)": f"On ({mu_val})" if mu_val > 0 else "Off (0.0)",
            "Differential Privacy": f"On (eps={max_eps_spent:.2f})" if use_dp_flag else "Off",
            "Accuracy (%)": f"{acc:.2f}%",
            "Macro F1 Score": round(macro_f1, 4)
        })

    # 6. Save Results to CSV
    results_df = pd.DataFrame(summary_rows)
    csv_path = './results/logs/ablation_results.csv'
    results_df.to_csv(csv_path, index=False)
    
    print("\n===========================================================================")
    print("📊 EDGESECAI ABLATION STUDY - FINAL RESULTS")
    print("===========================================================================")
    print(results_df.to_markdown(index=False))
    print(f"\n✅ Results saved to {csv_path}")

if __name__ == "__main__":
    main()