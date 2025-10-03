## Problem statement & goal
Develop a reproducible deep learning nowcasting pipeline matching Fabel et al. (2023/2024) that fuses all-sky imagery and recent irradiance measurements to predict short-term global horizontal irradiance (GHI). The objective is to deliver robust 1–20 minute ahead forecasts that outperform smart persistence across diverse sky conditions while remaining implementation-ready for later dataset- and code-focused steps.

## What we forecast (GHI) and the horizon (multi-step, 1–20 min ahead).
- Predict future GHI at minute-level resolution for lead times t ∈ {1, …, 20} minutes.
- Train a single multi-output regression head so the model jointly reasons about correlations between horizons.
- During inference expose the full 20-step vector; downstream consumers can optionally aggregate (e.g., rolling mean) to match operational cadence.

## Why all-sky imagers + recent irradiance time series are used.
- All-sky imagers capture imminent cloud motion, optical thickness, and horizon dynamics unavailable from point sensors alone.
- Recent irradiance and solar position sequences encode instantaneous atmospheric state and clear-sky deviations, stabilizing predictions under changing lighting or occlusions.
- Combining both sources enables the transformer to couple spatial cloud evolution with temporal intensity trends, closing the gap between purely physical persistence models and deep learning approaches.

## Inputs & targets (with shapes)
- ASI clip: `float32` tensor shaped `[T=5, C=3, H=128, W=128]`, sampled every minute over the last 5 minutes, already radiometrically calibrated and center-cropped to remove mounting artifacts.
- Time-series window: `float32` tensor shaped `[L=30, F=5]` comprising the past 30 minutes of clear-sky indices `k_GHI`, `k_DNI`, `k_DHI` plus `sun_elev_deg`, `sun_azim_deg` at 1-minute cadence.
- Targets: `float32` vector shaped `[20]` with the future GHI sequence (W/m²) for the next 1–20 minutes.

## Preprocessing & normalization
- Compute clear-sky indices `k = I_obs / I_clear` using validated clear-sky models; precompute sun elevation/azimuth from site metadata.
- Restrict training/eval samples to sun elevation ≥ 10°; optionally reweight or filter low-variability (near-constant GHI) windows to avoid degenerate gradients.
- Time-series features: fit mean/standard deviation on the training split and standardize per feature; persist stats for inference.
- Image frames: apply per-channel normalization using dataset-level means/stds; optionally mask sun occluder artifacts before normalization.
- Data augmentation: lightweight geometric ops (horizontal/vertical flips, ±15° rotations) applied consistently across the 5-frame clip; disable augmentations during validation/test.

## Model concept (high level)
- **Time-series branch:** Transformer encoder with learned positional encodings over 30 steps; final `[CLS]` or pooled token yields `z_ts ∈ ℝ^512`.
- **Image branch:** TimeSformer-style ViT with divided space-time attention and patch size (e.g., 16×16); 5 temporal tokens processed jointly to produce `z_asi ∈ ℝ^512`.
- **Fusion & head:** Concatenate embeddings `z = [z_ts || z_asi] ∈ ℝ^1024`, pass through a 2-layer MLP (hidden width 1024, GELU, dropout) to emit 20 regression outputs aligned with the forecast horizons.
- Train end-to-end from scratch; no branch-specific pretraining.

## Training setup (defaults to reproduce paper)
- Loss: mean squared error (MSE) summed/averaged across the 20 horizons.
- Optimizer: AdamW with `weight_decay=0.01` and decoupled weight decay on all parameters.
- Scheduler: One-Cycle policy with `max_lr=1e-3`, `pct_start≈0.1`, final cosine anneal to `max_lr/25`.
- Batch size 16, 10 epochs; gradient clipping (1.0) to stabilize early training.
- Mixed precision training (AMP) recommended to fit larger batches on GPUs while matching reported setup.
- Log metrics each epoch: RMSE, MAE, mean bias error (MBE) per horizon and averaged, plus forecast skill vs smart persistence baseline.

## Evaluation protocol
- Evaluate on a held-out test split with synchronized ASI + irradiance data; report per-horizon RMSE/MAE/MBE and aggregated statistics (mean and RMSE over full 20-min vector).
- Analyze temporal aggregation by averaging forecasts over 10–15 minute windows to mirror operational energy management horizons; compare against raw minute-wise outputs.
- Stratify results by DNI variability classes (e.g., stable, transitioning, highly variable) to replicate the paper’s robustness study and highlight gains over persistence in challenging regimes.
- Forecast skill vs smart persistence is the primary benchmark; compute skill for each lead time and aggregated window to demonstrate practical value.
