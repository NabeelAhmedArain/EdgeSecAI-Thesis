# =====================================================================
# EdgeSecAI: Generate 300-DPI Publication Plots for MS Thesis
# =====================================================================

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(summary_rows, edge_df=None, sweep_df=None, output_dir='./results/figures'):
    """
    Generates 300-DPI publication-ready plots for the thesis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Plot E1-E4 Ablation Bar Chart
    # ---------------------------------------------------------
    exp_labels = ["E1: FedAvg", "E2: FedProx", "E3: DP Only", "E4: EdgeSecAI\n(FedProx+DP)"]
    acc_vals = [float(r["Accuracy (%)"].replace("%", "")) for r in summary_rows]
    f1_vals = [float(r["Macro F1 Score"]) * 100 for r in summary_rows]  # Scaled to 0-100

    x = np.arange(len(exp_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy (%)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, f1_vals, width, label='Macro F1 (x100)', color='#3690c0')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('EdgeSecAI Ablation Study: Accuracy & Macro F1 (5 Clients, 3 Rounds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(40, 65)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    ablation_plot_path = os.path.join(output_dir, "Fig1_Ablation_Comparison.png")
    plt.savefig(ablation_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved Ablation Plot -> {ablation_plot_path}")

    # ---------------------------------------------------------
    # 2. Plot FP32 vs INT8 Edge Efficiency Comparison
    # ---------------------------------------------------------
    if edge_df is not None:
        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
        variants = ["FP32 Baseline", "INT8 Quantized (EdgeSecAI)"]
        sizes = edge_df["Size (MB)"].tolist()
        latencies = edge_df["Latency (ms/sample)"].tolist()

        x_e = np.arange(len(variants))
        b1 = ax1.bar(x_e - 0.18, sizes, 0.32, label='Model Size (MB)', color='#d95f02')
        b2 = ax1.bar(x_e + 0.18, latencies, 0.32, label='Latency (ms/sample)', color='#7570b3')

        ax1.set_ylabel('Magnitude (MB / ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Edge Deployment Efficiency: FP32 vs. Dynamic INT8 Quantization', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_e)
        ax1.set_xticklabels(variants, fontsize=10, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        for rect in b1 + b2:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        edge_plot_path = os.path.join(output_dir, "Fig2_Edge_Quantization_Efficiency.png")
        plt.savefig(edge_plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved Edge Efficiency Plot -> {edge_plot_path}")

    # ---------------------------------------------------------
    # 3. Plot Privacy-Utility Trade-off Curve
    # ---------------------------------------------------------
    if sweep_df is not None:
        fig, ax2 = plt.subplots(figsize=(7, 4.5), dpi=300)
        eps_x = [str(e) for e in sweep_df["Target Epsilon (ε)"]]
        f1_y = sweep_df["Macro F1 Score"].tolist()
        acc_y = [a / 100.0 for a in sweep_df["Accuracy (%)"]]

        ax2.plot(eps_x, f1_y, marker='o', linewidth=2.5, markersize=8, label='Macro F1 Score', color='#1b9e77')
        ax2.plot(eps_x, acc_y, marker='s', linewidth=2.0, linestyle='--', markersize=7, label='Accuracy (Normalized)', color='#e7298a')

        ax2.set_xlabel('Privacy Budget (Target Epsilon ε)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax2.set_title('Privacy-Utility Trade-Off Across Epsilon Budgets (EdgeSecAI)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        sweep_plot_path = os.path.join(output_dir, "Fig3_Privacy_Utility_Tradeoff.png")
        plt.savefig(sweep_plot_path, dpi=300)
        plt.close()
        print(f"✅ Saved Privacy-Utility Plot -> {sweep_plot_path}")

if __name__ == "__main__":
    print("✅ Plotting Module Loaded. (Run via main script)")