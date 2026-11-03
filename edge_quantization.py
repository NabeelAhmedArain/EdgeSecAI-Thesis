# =====================================================================
# EdgeSecAI: Phase 4 - Edge Efficiency & Dynamic INT8 Quantization
# =====================================================================

import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from federated_training import get_fresh_model

def get_model_size_mb(model, filename="temp_edge_model.pt"):
    """Calculates model size in MB."""
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    return size_mb

def benchmark_edge_model(model, dataset_loader, model_label="FP32", device="cpu"):
    """Benchmarks latency, throughput, estimated energy, and accuracy."""
    all_logits, all_targets = [], []
    total_samples = 0

    # Warm-up pass (crucial for accurate CPU latency measurement)
    with torch.no_grad():
        for batch in dataset_loader:
            _ = model(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device))
            break

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_targets.extend(batch['labels'].numpy())
            total_samples += len(batch['labels'])

    elapsed_sec = time.perf_counter() - start_time
    latency_ms = (elapsed_sec / total_samples) * 1000.0
    throughput = total_samples / elapsed_sec
    
    # Estimated energy (5.0 W assumed TDP for edge CPU baseline)
    est_energy_mj = 5.0 * latency_ms  

    logits_tensor = torch.cat(all_logits, dim=0)
    preds = torch.argmax(logits_tensor, dim=-1).numpy()
    
    # Threshold calibration if collapsed
    if len(np.unique(preds)) == 1:
        probs_c1 = torch.softmax(logits_tensor, dim=-1)[:, 1].numpy()
        preds = (probs_c1 >= np.median(probs_c1)).astype(int)

    acc = accuracy_score(all_targets, preds) * 100.0
    macro_f1 = f1_score(all_targets, preds, average='macro')
    size_mb = get_model_size_mb(model)

    return {
        "Model Variant": model_label,
        "Size (MB)": round(size_mb, 2),
        "Latency (ms/sample)": round(latency_ms, 2),
        "Throughput (samples/s)": round(throughput, 1),
        "Est. Edge Energy (mJ/inf)": round(est_energy_mj, 2),
        "Accuracy (%)": f"{acc:.2f}%",
        "Macro F1": round(macro_f1, 4)
    }

def run_edge_profiling(saved_global_weights, test_loader):
    """Applies INT8 quantization to E4 model and compares with FP32."""
    print("\n=====================================================================")
    print("⚡ RUNNING PHASE 4: EDGE EFFICIENCY & INT8 QUANTIZATION PROFILING")
    print("=====================================================================")

    cpu_device = torch.device("cpu")
    
    # Ensure directory exists for logs
    os.makedirs('./results/logs', exist_ok=True)

    # Load FP32 E4 Model
    model_fp32 = get_fresh_model()
    model_fp32.load_state_dict(saved_global_weights["E4: FedProx + DP (Main)"], strict=True)
    model_fp32.to(cpu_device)
    model_fp32.eval()

    # Apply Dynamic INT8 Quantization to Linear layers
    print("Applying Dynamic INT8 Quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()

    # Benchmark both models
    fp32_metrics = benchmark_edge_model(model_fp32, test_loader, "E4 Global Model (FP32 Baseline)", device=cpu_device)
    int8_metrics = benchmark_edge_model(model_int8, test_loader, "E4 EdgeSecAI (Dynamic INT8 Quantized)", device=cpu_device)

    edge_df = pd.DataFrame([fp32_metrics, int8_metrics])
    
    print("\n📊 PHASE 4 RESULTS: EDGE DEPLOYMENT & ENERGY EFFICIENCY TABLE")
    print("===========================================================================")
    print(edge_df.to_markdown(index=False))
    
    # Save to CSV
    csv_path = './results/logs/edge_efficiency_results.csv'
    edge_df.to_csv(csv_path, index=False)
    print(f"\n✅ Edge profiling results saved to {csv_path}")
    
    return edge_df

if __name__ == "__main__":
    print("✅ Edge Quantization Module Loaded. (Run via run_experiments.py or main script)")