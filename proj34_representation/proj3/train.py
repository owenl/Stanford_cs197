#!/usr/bin/env python3
"""Week 5 deliverable (Project 3): one reproducible experiment script.

Trains the tabular VAE with ONE numerical encoding condition and evaluates the
learned representation with the Week 4 suite (downstream probe, per-feature
recovery, latent geometry). Only the encoding changes between runs.

Usage:
    python train.py --encoding quantile_binning --seed 42
    python train.py --encoding standard --seed 1 --fair-params
    python train.py --encoding ple --epochs 100 --latent-dim 8

Run from this folder (proj3/). Data paths are the course layout (../../data).
Outputs: <out>/metrics.json, <out>/model.pt, <out>/Z.npz
"""
import argparse, copy, json, pickle as pk, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (proj3 -> proj34_representation -> root)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from proj34_representation.tabular_vae.vae import TabularVAE, vae_loss
from proj34_representation.tabular_vae.preprocessing import TabularPreprocessor
from proj34_representation.tabular_vae import encoders as enc_mod

ENCODINGS = {
    "standard": enc_mod.StandardEncoder,
    "quantile": enc_mod.QuantileEncoder,
    "mode_specific": enc_mod.ModeSpecificEncoder,
    "quantile_binning": lambda: enc_mod.QuantileBinningEncoder(n_bins=10),
    "ple": lambda: enc_mod.QuantilePLEEncoder(n_bins=8),
    "yeojohnson": enc_mod.YeoJohnsonEncoder,
    "periodic": lambda: enc_mod.PeriodicEncoder(n_frequencies=4),
}
DATA = Path(__file__).resolve().parents[2] / "data"   # <project>/data (symlink in work copy)


def vae_param_count(input_dim, cat_dims, hidden, latent):
    m = TabularVAE(input_dim, cat_dims, hidden, latent)
    return sum(p.numel() for p in m.parameters())


def pick_hidden(input_dim, cat_dims, latent, target_params, lo=8, hi=512):
    """Hidden size whose VAE parameter count is closest to target (fairness)."""
    return min(range(lo, hi + 1), key=lambda h: abs(vae_param_count(input_dim, cat_dims, h, latent) - target_params))


def run_epoch(model, loader, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"recon_cat": 0.0, "recon_cont": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for (x,) in loader:
            x_hat, mu, logvar = model(x)
            loss, parts = vae_loss(x, x_hat, mu, logvar, model.cat_dims, beta)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            n += x.size(0)
            for k in sums: sums[k] += parts[k] * x.size(0)
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def get_mu(model, X):
    model.eval(); out = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=512):
        _, mu, _ = model(xb); out.append(mu.numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True, choices=list(ENCODINGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kl-warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fair-params", action="store_true",
                    help="adjust hidden_dim so #params matches the standard-encoding model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out or f"results_week5/{args.encoding}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    feature_sets = pk.load(open(DATA / "feature_names.pkl", "rb"))
    patient_agg = pd.read_parquet(DATA / "clean_dataset.parquet")
    xc = json.load(open(DATA / "x_cols.json"))
    X_cols = xc["all"]
    splits = json.load(open(DATA / "splits.json"))
    dfs = {k: patient_agg.iloc[v].copy() for k, v in splits.items()}
    num_cols = [c for c in X_cols if c in feature_sets["labs"].tolist()
                + feature_sets["vitals"].tolist() + ["age", "admissionheight", "admissionweight"]]
    cat_cols = [c for c in X_cols if c not in num_cols]

    medians = dfs["train"][num_cols].apply(
        lambda s: pd.to_numeric(s.replace("> 89", 90), errors="coerce").median())
    def impute(df):
        num = df[num_cols].copy()
        for c in num_cols:
            num[c] = pd.to_numeric(num[c].replace("> 89", 90), errors="coerce")
        cat = df[cat_cols].copy()
        for c in cat_cols: cat[c] = cat[c].fillna("Missing")
        return pd.concat([num.fillna(medians), cat], axis=1)[X_cols]
    imp = {k: impute(df) for k, df in dfs.items()}

    make = ENCODINGS[args.encoding]
    pre = TabularPreprocessor(cat_cols, num_cols,
                              cont_encoders={c: make() for c in num_cols})
    Xmat = {"train": pre.fit_transform(imp["train"])}
    for k in ["val", "test"]:
        keep = np.ones(len(imp[k]), dtype=bool)
        for c in cat_cols:
            keep &= imp[k][c].isin(pre.categories_[c]).to_numpy()
        imp[k] = imp[k][keep]
        Xmat[k] = pre.transform(imp[k])
    Y = {k: patient_agg.loc[imp[k].index, "mortality_at48h"].fillna(0).astype(int) for k in imp}

    hidden = args.hidden_dim
    if args.fair_params:
        target = vae_param_count(
            pre.cat_total_ + len(num_cols), pre.cat_dims_, args.hidden_dim, args.latent_dim)
        hidden = pick_hidden(pre.input_dim_, pre.cat_dims_, args.latent_dim, target)
        print(f"[fair-params] target={target}, input_dim={pre.input_dim_}, hidden {args.hidden_dim} -> {hidden}")

    model = TabularVAE(pre.input_dim_, pre.cat_dims_, hidden, args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ltr = DataLoader(TensorDataset(torch.from_numpy(Xmat["train"])), args.batch_size, shuffle=True)
    lva = DataLoader(TensorDataset(torch.from_numpy(Xmat["val"])), args.batch_size, shuffle=False)
    best = (float("inf"), None); hist = []
    for ep in range(1, args.epochs + 1):
        b = args.beta * min(1.0, ep / max(1, args.kl_warmup))
        run_epoch(model, ltr, b, opt)
        vp = run_epoch(model, lva, b)
        vl = vp["recon_cat"] + vp["recon_cont"] + args.beta * vp["kl"]
        hist.append({"epoch": ep, "val_loss": vl, **vp})
        if vl < best[0]: best = (vl, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    Z = {k: get_mu(model, Xmat[k]) for k in Xmat}
    np.savez(out / "Z.npz", **{f"Z_{k}": Z[k] for k in Z})

    # Week 4 evaluation suite
    probe = LogisticRegression(max_iter=1000).fit(Z["train"], Y["train"])
    pz = probe.predict_proba(Z["test"])[:, 1]
    rec = {}
    for c in num_cols:
        lr = LinearRegression().fit(Z["train"], imp["train"][c])
        rec[c] = float(r2_score(imp["test"][c], lr.predict(Z["test"])))
    sc = StandardScaler().fit(imp["train"][num_cols])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Z["test"]), size=min(300, len(Z["test"])), replace=False)
    def pw(A):
        d = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)); return d[np.triu_indices(len(A), 1)]
    dX, dZ = pw(sc.transform(imp["test"][num_cols])[idx]), pw(Z["test"][idx])

    metrics = {
        "encoding": args.encoding, "seed": args.seed, "latent_dim": args.latent_dim,
        "hidden_dim": hidden, "input_dim": int(pre.input_dim_), "n_params": int(n_params),
        "best_val_loss": float(best[0]),
        "final_val_parts": {k: float(v) for k, v in run_epoch(model, lva, args.beta).items()},
        "probe_test_AUROC": float(roc_auc_score(Y["test"], pz)),
        "probe_test_AUPRC": float(average_precision_score(Y["test"], pz)),
        "recovery_R2": rec, "mean_recovery_R2": float(np.mean(list(rec.values()))),
        "geometry_pearson": float(pearsonr(dX, dZ)[0]),
        "geometry_spearman": float(spearmanr(dX, dZ)[0]),
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps({k: metrics[k] for k in
                      ["encoding", "seed", "input_dim", "n_params", "best_val_loss",
                       "probe_test_AUROC", "probe_test_AUPRC", "mean_recovery_R2",
                       "geometry_spearman"]}, indent=1))
    print("Wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
