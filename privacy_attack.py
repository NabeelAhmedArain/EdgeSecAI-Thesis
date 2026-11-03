# =====================================================================
# EdgeSecAI: Phase 4 - Gradient Inversion Attack (Privacy Validation)
# =====================================================================

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rouge_score import rouge_scorer
from federated_training import device

def get_target_gradients(model, input_ids, attention_mask, labels, simulate_dp_noise=False):
    """Extracts true gradients from a given model (optionally adding simulated DP noise)."""
    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    
    grads = []
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                g = param.grad.clone()
                if simulate_dp_noise:
                    # Simulate DP clipping and noise (if not already done by the model itself)
                    norm = torch.norm(g)
                    clip_factor = min(1.0, 1.0 / (norm + 1e-6))
                    g = g * clip_factor
                    g += torch.randn_like(g) * 0.5
                grads.append(g)
            else:
                grads.append(torch.zeros_like(param))
    model.zero_grad()
    return grads

def run_dlg_attack(model, target_grads, input_ids, labels, tokenizer, steps=300, lr=0.1):
    """Runs the DLG attack optimization to reconstruct text."""
    seq_len = input_ids.shape[1]
    embed_dim = model.distilbert.embeddings.word_embeddings.embedding_dim
    
    # Initialize with actual vocab embeddings instead of pure noise
    vocab_size = model.distilbert.embeddings.word_embeddings.weight.shape[0]
    random_indices = torch.randint(0, vocab_size, (1, seq_len), device=device)
    initial_embeds = model.distilbert.embeddings.word_embeddings.weight[random_indices].clone().detach()
    
    dummy_embeddings = initial_embeds.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_embeddings], lr=lr)
    params = [p for p in model.parameters() if p.requires_grad]

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs_embeds=dummy_embeddings, labels=labels)
        dummy_grads = torch.autograd.grad(outputs.loss, params, create_graph=True, allow_unused=True)
        
        loss = 0
        for dg, tg in zip(dummy_grads, target_grads):
            if dg is not None:
                loss += (1 - F.cosine_similarity(dg.view(-1), tg.view(-1), dim=0))
                
        loss.backward()
        optimizer.step()

    # Reconstruct text from optimized embeddings
    dummy_norm = F.normalize(dummy_embeddings, dim=-1)
    vocab_norm = F.normalize(model.distilbert.embeddings.word_embeddings.weight, dim=-1)
    similarity = torch.matmul(dummy_norm, vocab_norm.T)
    recovered_ids = torch.argmax(similarity, dim=-1).squeeze(0)
    recovered_text = tokenizer.decode(recovered_ids, skip_special_tokens=True)
    return recovered_text, recovered_ids

def evaluate_privacy_attack(saved_global_weights, test_texts):
    """Runs the privacy attack against E1, E3, and E4 models."""
    print("\n=====================================================================")
    print("🛡️ RUNNING PHASE 4: PRIVACY VALIDATION (GRADIENT INVERSION ATTACK)")
    print("=====================================================================")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 1. Select the shortest text to ensure attack success
    short_texts = [t for t in test_texts if 5 <= len(t.split()) <= 15]
    sample_text = short_texts[0] if short_texts else "this is extremely harmful hate speech"
    
    print(f"Target Original Text: '{sample_text}'\n")
    
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True, max_length=20)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = torch.tensor([1], dtype=torch.long).to(device) # Force label 1

    attack_results = []
    experiments = [
        ("E1: Baseline (No DP)", "E1: Baseline (FedAvg)", False),
        ("E3: DP Only", "E3: DP Only", True),
        ("E4: FedProx + DP (Main)", "E4: FedProx + DP (Main)", True)
    ]

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    orig_ids = [i for i in input_ids.squeeze(0).cpu().tolist() if i != tokenizer.pad_token_id]

    for display_name, model_key, simulate_noise in experiments:
        if model_key not in saved_global_weights:
            print(f"⚠️ Weights for {model_key} not found. Skipping.")
            continue

        print(f"--- Running Attack on {display_name} ---")
        
        # Load specific model weights
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, attn_implementation="eager"
        ).to(device)
        model.load_state_dict(saved_global_weights[model_key], strict=True)
        model.eval()

        # Extract target gradients
        target_grads = get_target_gradients(model, input_ids, attention_mask, labels, simulate_dp_noise=simulate_noise)

        # Run attack
        rec_text, rec_ids = run_dlg_attack(model, target_grads, input_ids, labels, tokenizer)

        # Calculate Metrics
        rouge_score = scorer.score(sample_text, rec_text)['rougeL'].fmeasure * 100
        match_rate = (sum(1 for o, r in zip(orig_ids, rec_ids.cpu().tolist()) if o == r) / len(orig_ids)) * 100 if orig_ids else 0.0

        attack_results.append({
            "Configuration": display_name,
            "Recovered Text": rec_text,
            "Token Match (%)": round(match_rate, 2),
            "ROUGE-L (%)": round(rouge_score, 2)
        })
        
        print(f"Recovered: '{rec_text}'")
        print(f"Metrics -> Token Match: {match_rate:.2f}% | ROUGE-L: {rouge_score:.2f}%\n")

    # Save Results
    results_df = pd.DataFrame(attack_results)
    csv_path = './results/logs/privacy_attack_results.csv'
    results_df.to_csv(csv_path, index=False)
    
    print("\n===========================================================================")
    print("📊 PRIVACY VALIDATION RESULTS TABLE")
    print("===========================================================================")
    print(results_df[["Configuration", "Token Match (%)", "ROUGE-L (%)"]].to_markdown(index=False))
    print(f"\n✅ Attack results saved to {csv_path}")

    return results_df

if __name__ == "__main__":
    print("✅ Privacy Attack Module Loaded. (Run via main script)")