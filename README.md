# EdgeSecAI: An Energy-Efficient and Privacy-Aware Federated Learning Framework for Text-Based Content Moderation

**Author:** Nabeel Ahmed  
**Degree:** MS Information Security  
**University:** NED University of Engineering & Technology  
**Supervisor:** Dr. Abdul Karim

## 📌 Overview
EdgeSecAI is a Federated Learning (FL) framework designed for edge-based AI content moderation. It addresses three critical challenges: privacy leakage (via Gradient Inversion Attacks), non-IID data distribution (via FedProx), and edge deployment efficiency (via INT8 Quantization).

## 🛠️ Core Technologies
- **Model:** DistilBERT-base-uncased
- **Federated Learning:** Flower (flwr)
- **Differential Privacy:** Opacus (DP-SGD, ε = 1.0)
- **Dataset:** HatEval (SemEval-2019 Task 5) & OLID

## 📊 Key Results (From Thesis Evaluation)
- **EdgeSecAI (E4) Macro F1:** 0.5596 (Accuracy: 56.26%)
- **Privacy Defense:** Reduced Gradient Inversion ROUGE-L score from 7.69% (No DP) to 0.00% (With DP).
- **Edge Efficiency:** Dynamic INT8 quantization reduced model size by 48.2% (255 MB to 132 MB) with only a 1% drop in accuracy.

## 📂 Repository Structure
- `data_loader.py`: Scripts for loading HatEval/OLID and creating Dirichlet client splits.
- `federated_training.py`: Core FedAvg/FedProx training loop with Opacus DP-SGD.
- `privacy_attack.py`: Deep Leakage from Gradients (DLG) attack implementation.
- `edge_quantization.py`: Dynamic INT8 quantization and latency profiling.
- `centralized_baseline.py`: Centralized DistilBERT training for upper-bound comparison.
- `epsilon_sweep.py`: Privacy-utility trade-off evaluation.
- `plotting.py`: Generates 300-DPI publication-ready figures.
- `main.py`: Master script to reproduce the entire pipeline.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/EdgeSecAI-MS-Thesis.git# EdgeSecAI-Thesis
Energy-Efficient and Privacy-Aware Federated Learning Framework for Edge-Based Content Moderation.

2. Install Dependencies
   pip install -r requirements.txt

3. Download the HatEval and OLID datasets and place them in a ./data folder.

4. Run the full pipeline:
   python main.py
