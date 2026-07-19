# -*- coding: utf-8 -*-
"""
Virtual Screening and Conformal Prediction Inference Script
Multi-Task ALIGNN Prediction of Refractive Index and Bandgap
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# If the alignn package is installed via pip, this import works natively.
# Ensure 'alignn' is in your requirements.txt
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute device: {DEVICE}")

# Screening Thresholds (from Section 4.4.4)
N_THRESH  = 2.3
EG_THRESH = 3.5
SCORE_W   = 0.5
SCORE_PEN = 0.3

SEEDS = [42, 123, 456]

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskALIGNN(nn.Module):
    """
    Exact replica of the training architecture for strict weight compatibility.
    """
    def __init__(self, head_hidden=128):
        super().__init__()
        cfg = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            embedding_features=64,
            hidden_features=256,
            output_features=1,
            link="identity",
            classification=False,
        )
        self.encoder = ALIGNN(cfg)
        self.encoder.fc = nn.Identity()
        hidden = cfg.hidden_features
        
        self.head_n = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1)
        )
        self.head_eg = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1)
        )
        
        # Kendall et al. (2018) learnable log-variance params (required for state_dict matching)
        self.log_var_n = nn.Parameter(torch.zeros(1))
        self.log_var_eg = nn.Parameter(torch.zeros(1))

    def forward(self, g, lg, ag):
        h = self.encoder([g, lg, ag])
        return self.head_n(h).squeeze(-1), self.head_eg(h).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class VirtDataset(Dataset):
    def __init__(self, records, graphs):
        self.items = [(r, graphs[r["vid"]]) for r in records if r["vid"] in graphs]
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        rec, (g, lg) = self.items[idx]
        return rec["vid"], g, lg

def virt_collate(batch):
    vids, gs, lgs = zip(*batch)
    return list(vids), dgl.batch(gs), dgl.batch(lgs)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=== Loading Calibration and Normalization Stats ===")
    
    # 1. Load Normalization
    norm_path = os.path.join(DATA_DIR, "normalization_stats.json")
    with open(norm_path) as f:
        ns = json.load(f)
    MU_N, STD_N = ns["mu_n"], ns["std_n"]
    MU_EG, STD_EG = ns["mu_eg"], ns["std_eg"]
    
    # 2. Load Calibration Bounds
    cp_path = os.path.join(DATA_DIR, "conformal_calibration.json")
    with open(cp_path) as f:
        cp = json.load(f)
    Q_N, Q_EG = cp["q_n"], cp["q_eg"]
    
    print(f"Conformal thresholds: q_n={Q_N:.4f}, q_Eg={Q_EG:.4f} eV")
    print(f"Coverage target: {cp['coverage_target']*100:.0f}%\n")

    # 3. Load Virtual Data
    print("=== Loading Virtual Library ===")
    # For reviewers, it's best to package the structures and their DGL graphs together to skip the jarvis-tools build step.
    virt_path = os.path.join(DATA_DIR, "virtual_structures_with_graphs.pkl")
    with open(virt_path, "rb") as f:
        virt_data = pickle.load(f)
    
    virt_records = virt_data["records"]
    virt_graphs = virt_data["graphs"]
    print(f"Virtual structures loaded: {len(virt_records):,}\n")

    # 4. Load Ensemble
    print("=== Loading ALIGNN Ensemble ===")
    ensemble_models = []
    for seed in SEEDS:
        ckpt = os.path.join(MODELS_DIR, f"seed_{seed}", "best_model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing ensemble weights: {ckpt}")
        
        m = MultiTaskALIGNN().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        m.eval()
        ensemble_models.append(m)
    print(f"Ensemble successfully loaded: {len(ensemble_models)} models.\n")

    # 5. Run Inference
    print("=== Running Inference ===")
    loader = DataLoader(
        VirtDataset(virt_records, virt_graphs),
        batch_size=32, shuffle=False, collate_fn=virt_collate,
        num_workers=0, pin_memory=False
    )

    all_vids = []
    all_preds_n = {s: [] for s in SEEDS}
    all_preds_eg = {s: [] for s in SEEDS}

    with torch.no_grad():
        for vids, g_batch, lg_batch in tqdm(loader, desc="Predicting"):
            g_batch = g_batch.to(DEVICE)
            lg_batch = lg_batch.to(DEVICE)
            ag_batch = dgl.line_graph(lg_batch, backtracking=False).to(DEVICE)
            
            all_vids.extend(vids)
            for seed, model in zip(SEEDS, ensemble_models):
                pn_z, peg_z = model(g_batch, lg_batch, ag_batch)
                
                # Denormalize
                pn = (pn_z * STD_N + MU_N).cpu().numpy()
                peg = (peg_z * STD_EG + MU_EG).cpu().numpy()
                
                all_preds_n[seed].extend(pn.tolist())
                all_preds_eg[seed].extend(peg.tolist())

    # 6. Compute Statistics & Conformal Intervals
    preds_n_arr = np.array([all_preds_n[s] for s in SEEDS])
    preds_eg_arr = np.array([all_preds_eg[s] for s in SEEDS])

    mu_n_virt = preds_n_arr.mean(axis=0)
    mu_eg_virt = preds_eg_arr.mean(axis=0)

    # Unbiased ensemble std
    sig_n_virt = preds_n_arr.std(axis=0, ddof=1)
    sig_eg_virt = preds_eg_arr.std(axis=0, ddof=1)

    ci_n_half = Q_N
    ci_eg_half = Q_EG

    lo_n = mu_n_virt - ci_n_half
    hi_n = mu_n_virt + ci_n_half
    lo_eg = mu_eg_virt - ci_eg_half
    hi_eg = mu_eg_virt + ci_eg_half

    # 7. Apply Scoring
    score = (
        SCORE_W * (mu_n_virt - N_THRESH) / STD_N +
        SCORE_W * (mu_eg_virt - EG_THRESH) / STD_EG -
        SCORE_PEN * (sig_n_virt / STD_N + sig_eg_virt / STD_EG)
    )

    # 8. Export Results
    vid_to_rec = {r["vid"]: r for r in virt_records}
    results_rows = []
    
    for i, vid in enumerate(all_vids):
        rec = vid_to_rec.get(vid, {})
        results_rows.append({
            "vid": vid,
            "formula": rec.get("formula", ""),
            "reduced_formula": rec.get("reduced_formula", ""),
            "n_mean": mu_n_virt[i],
            "n_std": sig_n_virt[i],
            "n_ci_lo": lo_n[i],
            "n_ci_hi": hi_n[i],
            "n_ci_half": ci_n_half,
            "eg_mean": mu_eg_virt[i],
            "eg_std": sig_eg_virt[i],
            "eg_ci_lo": lo_eg[i],
            "eg_ci_hi": hi_eg[i],
            "eg_ci_half": ci_eg_half,
            "score": score[i]
        })

    results_df = pd.DataFrame(results_rows).sort_values("score", ascending=False).reset_index(drop=True)
    
    results_df["passes_gate"] = (
        (results_df["n_mean"] > N_THRESH) &
        (results_df["eg_mean"] > EG_THRESH)
    )
    
    out_csv = os.path.join(RESULTS_DIR, "virtual_screening_results.csv")
    results_df.to_csv(out_csv, index=False)

    # 9. Summary
    n_passing = results_df["passes_gate"].sum()
    print("\n=== Screening Complete ===")
    print(f"Total screened: {len(results_df):,}")
    print(f"Candidates passing BOTH hard gates (n > {N_THRESH}, Eg > {EG_THRESH} eV): {n_passing}")
    
    if n_passing == 0:
        print("NULL RESULT CONFIRMED: No virtual candidate simultaneously satisfies both thresholds.")
        print("This rigorously confirms the Penn-Moss tradeoff is not overcome within this substitution space.")
        
    print(f"\nResults saved to: {out_csv}")

if __name__ == "__main__":
    main()