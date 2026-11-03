# =====================================================================
# EdgeSecAI: Phase 3 - Privacy-Utility Trade-off (Epsilon Sweep)
# =====================================================================

import copy
import pandas as pd
import torch
from federated_training import train_client_edgesecai, weighted_fedavg
from evaluation import evaluate_global_model

def run_epsilon_sweep(train_dataset, clients_map, initial_base_state, test_loader, e4_baseline_metrics=None):
    """
    Runs the Epsilon Sweep for the E4 (FedProx + DP) configuration.
    If e4_baseline_metrics (acc, f1) is provided for eps=1.0, it reuses it to save time.
    """
    print("\n=====================================================================")
    print("🔒 RUNNING PHASE 3: PRIVACY-UTILITY TRADE-OFF (EPSILON SWEEP)")
    print("=====================================================================")

    epsilon_list = [0.5, 1.0, 3.0, 8.0]
    sweep_rows = []

    # Sweep Configuration (using fewer clients/rounds for faster execution)
    SWEEP_CLIENTS = 3
    SWEEP_ROUNDS = 2
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    MU = 0.01

    for target_eps in epsilon_list:
        # Optimization: Reuse the main E4 result for eps=1.0 if provided
        if target_eps == 1.0 and e4_baseline_metrics is not None:
            e4_acc_val, e4_f1_val = e4_baseline_metrics
            print(f"\n✅ Reusing completed E4 result for Epsilon = 1.0 -> Acc: {e4_acc_val:.2f}%, F1: {e4_f1_val:.4f}")
            sweep_rows.append({
                "Target Epsilon (ε)": target_eps,
                "Privacy Level": "Strict (Main E4)",
                "Accuracy (%)": round(e4_acc_val, 2),
                "Macro F1 Score": round(e4_f1_val, 4)
            })
            continue

        print(f"\nRunning EdgeSecAI (FedProx + DP) for Target Epsilon = {target_eps}...")
        
        sweep_state = copy.deepcopy(initial_base_state)
        
        for r in range(1, SWEEP_ROUNDS + 1):
            r_states, r_sizes = [], []
            for cid in list(clients_map.keys())[:SWEEP_CLIENTS]:
                c_indices = clients_map[cid]
                c_state, _ = train_client_edgesecai(
                    global_state_dict=sweep_state,
                    dataset=train_dataset,
                    indices=c_indices,
                    epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    use_dp=True,
                    epsilon=target_eps,
                    delta=1e-5,
                    mu=MU
                )
                r_states.append(c_state)
                r_sizes.append(len(c_indices))
            sweep_state = weighted_fedavg(r_states, r_sizes)

        s_acc, s_f1, _ = evaluate_global_model(sweep_state, test_loader, exp_name=f"E4 (eps={target_eps})")
        
        level_label = "Ultra-Strict" if target_eps < 1.0 else ("Moderate" if target_eps == 3.0 else "Relaxed")
        sweep_rows.append({
            "Target Epsilon (ε)": target_eps,
            "Privacy Level": level_label,
            "Accuracy (%)": round(s_acc, 2),
            "Macro F1 Score": round(s_f1, 4)
        })

    sweep_df = pd.DataFrame(sweep_rows)
    
    print("\n📊 PHASE 3 RESULTS: PRIVACY-UTILITY TRADE-OFF TABLE")
    print("===========================================================================")
    print(sweep_df.to_markdown(index=False))
    
    # Save results to CSV
    csv_path = './results/logs/epsilon_sweep_results.csv'
    sweep_df.to_csv(csv_path, index=False)
    print(f"\n✅ Epsilon sweep results saved to {csv_path}")

    return sweep_df

if __name__ == "__main__":
    print("✅ Epsilon Sweep Module Loaded. (Run via main script)")