# =====================================================================
# EdgeSecAI: Master Execution Script (main.py)
# =====================================================================
import os
import copy
import pandas as pd
import torch
from collections import OrderedDict

# Import custom modules
from data_loader import load_and_prepare_data, create_dirichlet_split
from evaluation import load_test_data, evaluate_global_model
from federated_training import get_fresh_model, train_client_edgesecai, weighted_fedavg
from centralized_baseline import train_centralized_baseline
from epsilon_sweep import run_epsilon_sweep
from edge_quantization import run_edge_profiling
from privacy_attack import evaluate_privacy_attack
from plotting import generate_plots

def main():
    print("=====================================================================")
    print("🚀 STARTING EDGESECAI FULL PIPELINE REPRODUCTION")
    print("=====================================================================")
    
    # 1. Setup Directories
    os.makedirs('./results/logs', exist_ok=True)
    os.makedirs('./results/figures', exist_ok=True)
    
    # 2. Load Datasets
    print("\n--- Step 1: Loading Datasets ---")
    _, _, _, train_dataset = load_and_prepare_data(data_dir='./data')
    test_loader = load_test_data(data_dir='./data')
    
    # Load test texts for the attack module
    test_df = pd.read_csv('./data/hateval_test.csv')
    test_texts = test_df['text'].astype(str).tolist()
    
    # 3. Create Client Partitions
    print("\n--- Step 2: Creating Federated Client Partitions ---")
    clients_map = create_dirichlet_split(train_dataset, num_clients=5, alpha=0.5)
    
    # Save initial base state for experiments
    torch.manual_seed(42)
    initial_base_state = OrderedDict({k: v.cpu() for k, v in get_fresh_model().state_dict().items()})

    # 4. Centralized Baseline
    print("\n--- Step 3: Training Centralized Baseline (Upper Bound) ---")
    central_acc, central_f1, _ = train_centralized_baseline(train_dataset, test_loader, epochs=2)

    # 5. Federated Ablation Study (E1 to E4)
    print("\n--- Step 4: Running Federated Ablation Study (E1-E4) ---")
    experiments_config = [
        ("E1: Baseline (FedAvg)",   False, 0.0,  0.0),
        ("E2: FedProx Only",        False, 0.01, 0.0),
        ("E3: DP Only",             True,  0.0,  1.0),
        ("E4: FedProx + DP (Main)", True,  0.01, 1.0),
    ]
    
    saved_global_weights = {}
    summary_rows = []
    NUM_ROUNDS = 3 # Note: Change to 10-15 for final full run

    for exp_title, use_dp_flag, mu_val, eps_val in experiments_config:
        print(f"\n🚀 STARTING {exp_title} | DP={use_dp_flag} | mu={mu_val}")
        global_state = copy.deepcopy(initial_base_state)
        max_eps_spent = 0.0

        for r in range(1, NUM_ROUNDS + 1):
            round_states, round_sizes = [], []
            for cid in clients_map.keys():
                c_state, c_eps = train_client_edgesecai(
                    global_state_dict=global_state,
                    dataset=train_dataset,
                    indices=clients_map[cid],
                    epochs=2, batch_size=32,
                    use_dp=use_dp_flag, epsilon=eps_val, delta=1e-5, mu=mu_val
                )
                round_states.append(c_state)
                round_sizes.append(len(clients_map[cid]))
                max_eps_spent = max(max_eps_spent, c_eps)
            global_state = weighted_fedavg(round_states, round_sizes)

        acc, macro_f1, _ = evaluate_global_model(global_state, test_loader, exp_name=exp_title)
        saved_global_weights[exp_title] = copy.deepcopy(global_state)
        
        summary_rows.append({
            "Experiment": exp_title,
            "FedProx (mu)": f"On ({mu_val})" if mu_val > 0 else "Off (0.0)",
            "Differential Privacy": f"On (eps={max_eps_spent:.2f})" if use_dp_flag else "Off",
            "Accuracy (%)": f"{acc:.2f}%",
            "Macro F1 Score": round(macro_f1, 4)
        })

    # Save Ablation to CSV
    ablation_df = pd.DataFrame(summary_rows)
    ablation_df.to_csv('./results/logs/ablation_results.csv', index=False)
    print("\n📊 EDGESECAI ABLATION STUDY RESULTS")
    print(ablation_df.to_markdown(index=False))

    # 6. Epsilon Sweep
    print("\n--- Step 5: Running Epsilon Sweep ---")
    # Get E4 accuracy and F1 to reuse for eps=1.0
    e4_row = next(item for item in summary_rows if "E4" in item["Experiment"])
    e4_baseline_metrics = (float(e4_row["Accuracy (%)"].replace("%", "")), float(e4_row["Macro F1 Score"]))
    
    sweep_df = run_epsilon_sweep(
        train_dataset, clients_map, initial_base_state, test_loader, e4_baseline_metrics
    )

    # 7. Edge Quantization Profiling
    print("\n--- Step 6: Running Edge Quantization Profiling ---")
    edge_df = run_edge_profiling(saved_global_weights, test_loader)

    # 8. Privacy Attack (Gradient Inversion)
    print("\n--- Step 7: Running Gradient Inversion Attack ---")
    attack_df = evaluate_privacy_attack(saved_global_weights, test_texts)

    # 9. Generate Publication Plots
    print("\n--- Step 8: Generating Publication Plots ---")
    generate_plots(summary_rows, edge_df, sweep_df, output_dir='./results/figures')

    print("\n=====================================================================")
    print("🎉 PIPELINE COMPLETE! Check the './results/' folder for all logs and figures.")
    print("=====================================================================")

if __name__ == "__main__":
    main()