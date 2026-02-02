# @title Step 1: ING-Net Ultimate (V4.2: Parallelized Simulation)
# @markdown **Version:** V4.2 (Parallel & Focused)
# @markdown **Change:** Implemented parallel data simulation (`num_workers`) to dramatically speed up training.

import os
import sys
import numpy as np
import torch
import time
from scipy.stats import kurtosis, pearsonr, iqr, gmean
from scipy.signal import periodogram
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm.auto import tqdm
from gwpy.timeseries import TimeSeries
import warnings

warnings.filterwarnings("ignore")

# --- Setup ---
try:
    import sbi
except ImportError:
    import os
    os.system(f"{sys.executable} -m pip install sbi gwpy corner -q")
    import sbi

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Processing on: {device.upper()}")

CACHE_DIR = "LIGO_Data_Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Using a new version number to save results separately
SAVE_FILE = os.path.join(CACHE_DIR, "ing_net_ultimate_results_v4.2.pt")
MODEL_DIR = os.path.join(CACHE_DIR, "models_v4.2")

# --- Config ---
DATASETS_CONFIG = {
    "O3a_WeakSignalFocus": 1238166018,
}
DURATION = 4096
TARGET_FS = 2048.0
LOW_FREQ = 100
HIGH_FREQ = 400
VETO_THRESH = 3.5
N_REPEATS = 3
N_SIMS = 50000

# --- Simulator and Eval functions remain unchanged ---
def build_enhanced_simulator(h1_bg, l1_bg, veto_threshold=3.0):
    # (Code is identical to previous version, so it's omitted for brevity)
    seg_len = int(4.0 * TARGET_FS)
    max_idx = len(h1_bg) - seg_len - 1
    def simulator(theta_batch):
        if isinstance(theta_batch, torch.Tensor): theta_batch = theta_batch.detach().cpu().numpy()
        batch = []
        for theta in theta_batch:
            log_omega, xi = float(theta[0]), float(theta[1])
            max_retries = 20
            for attempt in range(max_retries):
                start = np.random.randint(0, max_idx)
                n_h1 = np.array(h1_bg[start : start+seg_len], dtype=np.float64)
                n_l1 = np.array(l1_bg[start : start+seg_len], dtype=np.float64)
                if np.abs(kurtosis(n_h1)) < veto_threshold and np.abs(kurtosis(n_l1)) < veto_threshold:
                    break
            med_h, scale_h = np.median(n_h1), iqr(n_h1)/1.349
            med_l, scale_l = np.median(n_l1), iqr(n_l1)/1.349
            n_h1 = (n_h1 - med_h) / (scale_h + 1e-15)
            n_l1 = (n_l1 - med_l) / (scale_l + 1e-15)
            omega = 10**log_omega
            safe_xi = np.max([xi, 1e-4])
            amp = np.sqrt(omega / safe_xi) * 50.0
            n_ev = int(seg_len * safe_xi * 0.2)
            sig = np.zeros(seg_len)
            if n_ev > 0:
                idx = np.random.randint(0, seg_len, n_ev)
                sig[idx] += np.random.normal(0, amp, n_ev)
            d_h1 = n_h1 + sig
            d_l1 = n_l1 + sig
            d_h1 = np.asarray(d_h1, dtype=np.float64)
            d_l1 = np.asarray(d_l1, dtype=np.float64)
            cc, _ = pearsonr(d_h1, d_l1)
            pw = np.log10(np.var(d_h1)*np.var(d_l1) + 1e-30)
            k_h = np.log1p(np.abs(kurtosis(d_h1)))
            k_l = np.log1p(np.abs(kurtosis(d_l1)))
            k_cross = np.log1p(np.abs(kurtosis(d_h1 * d_l1)))
            _, Pxx_h = periodogram(d_h1, fs=TARGET_FS)
            _, Pxx_l = periodogram(d_l1, fs=TARGET_FS)
            sfm_h = gmean(Pxx_h + 1e-20) / (np.mean(Pxx_h) + 1e-20)
            sfm_l = gmean(Pxx_l + 1e-20) / (np.mean(Pxx_l) + 1e-20)
            n_blocks = 64
            len_trim = (len(d_h1) // n_blocks) * n_blocks
            blocks_h = d_h1[:len_trim].reshape(n_blocks, -1)
            blocks_l = d_l1[:len_trim].reshape(n_blocks, -1)
            max_pw_h = np.log10(np.max(np.sum(blocks_h**2, axis=1)) + 1e-30)
            max_pw_h = np.clip(max_pw_h, -50, 50)
            max_pw_l = np.log10(np.max(np.sum(blocks_l**2, axis=1)) + 1e-30)
            max_pw_l = np.clip(max_pw_l, -50, 50)
            features = [cc, pw, k_h, k_l, k_cross, sfm_h, sfm_l, max_pw_h, max_pw_l]
            features = np.array(features) + np.random.normal(0, 0.05, size=len(features))
            batch.append(torch.tensor(features, dtype=torch.float32))
        return torch.stack(batch)
    return simulator

def calculate_limit(posterior, sim_fn, xi_target, method="ai", n_trials=20):
    # (Code is identical to previous version, so it's omitted for brevity)
    omega_scan = np.linspace(-3.0, -15.0, 40)
    thresh = -10.0
    for log_omega in omega_scan:
        theta_batch = torch.tensor([[log_omega, xi_target]] * n_trials)
        obs_batch = sim_fn(theta_batch).to(device)
        valid_idx = ~torch.isnan(obs_batch).any(dim=1)
        if valid_idx.sum() == 0: continue
        obs_batch = obs_batch[valid_idx]
        current_trials = obs_batch.shape[0]
        detected_count = 0
        for i in range(current_trials):
            if method == "ai":
                samps = posterior.sample((1000,), x=obs_batch[i], show_progress_bars=False)
            else:
                samps = posterior.sample((1000,), x=obs_batch[i, [0, 1]], show_progress_bars=False)
            if np.percentile(samps.cpu()[:, 0].numpy(), 5) > thresh:
                detected_count += 1
        if (detected_count / current_trials) < 0.70:
            return log_omega + (12.0/40.0)
    return -15.0

# --- Load Previous Results ---
if os.path.exists(SAVE_FILE):
    FINAL_RESULTS = torch.load(SAVE_FILE)
    print(f"✅ Resuming... Done: {list(FINAL_RESULTS.keys())}")
else:
    FINAL_RESULTS = {}

# --- Main Loop ---
xi_values = [0.001, 0.01, 0.1, 0.5, 1.0]
total_steps = len(DATASETS_CONFIG) * N_REPEATS
global_pbar = tqdm(total=total_steps, desc="Global Progress", unit="run")
start_time = time.time()

for label, gps in DATASETS_CONFIG.items():
    if label in FINAL_RESULTS:
        print(f"⏩ Skipping {label}.")
        global_pbar.update(N_REPEATS)
        continue
    tqdm.write(f"\n🌍 Processing {label} (V4.2 Parallel & Focused Mode)...")

    # --- Data Loading (identical) ---
    h1_cache_path = os.path.join(CACHE_DIR, f"{label.split('_')[0]}_H1.npy")
    l1_cache_path = os.path.join(CACHE_DIR, f"{label.split('_')[0]}_L1.npy")
    current_h1, current_l1 = None, None
    if gps is None:
        current_h1 = np.random.normal(0, 1, int(DURATION*TARGET_FS))
        current_l1 = np.random.normal(0, 1, int(DURATION*TARGET_FS))
    elif os.path.exists(h1_cache_path):
        current_h1 = np.load(h1_cache_path)
        current_l1 = np.load(l1_cache_path)
    else:
        try:
            tqdm.write(f"   ⬇️ Downloading {label.split('_')[0]}...")
            h1_ts = TimeSeries.fetch_open_data('H1', gps, gps+DURATION+32, verbose=False)
            current_h1 = h1_ts.resample(TARGET_FS).whiten(4, 2).bandpass(LOW_FREQ, HIGH_FREQ).crop(gps+4, gps+4+DURATION).value.copy()
            l1_ts = TimeSeries.fetch_open_data('L1', gps, gps+DURATION+32, verbose=False)
            current_l1 = l1_ts.resample(TARGET_FS).whiten(4, 2).bandpass(LOW_FREQ, HIGH_FREQ).crop(gps+4, gps+4+DURATION).value.copy()
            np.save(h1_cache_path, current_h1)
            np.save(l1_cache_path, current_l1)
        except Exception as e:
            tqdm.write(f"   ⚠️ Failed: {e}. Using Mock.")
            current_h1 = np.random.normal(0, 1, int(DURATION*TARGET_FS))
            current_l1 = np.random.normal(0, 1, int(DURATION*TARGET_FS))

    res_ai = np.zeros((N_REPEATS, len(xi_values)))
    res_tr = np.zeros((N_REPEATS, len(xi_values)))
    model_save_dir = os.path.join(MODEL_DIR, label)
    os.makedirs(model_save_dir, exist_ok=True)

    for r in range(N_REPEATS):
        elapsed = time.time() - start_time
        global_pbar.set_postfix_str(f"Run {r+1}/{N_REPEATS} | {label}")

        sim = build_enhanced_simulator(current_h1, current_l1, veto_threshold=VETO_THRESH)

        # --- Focused Prior Definition (identical) ---
        n_weak_sims = N_SIMS // 2
        n_broad_sims = N_SIMS - n_weak_sims
        prior_weak = BoxUniform(low=torch.tensor([-10.0, 0.001], device=device), high=torch.tensor([-5.0, 1.0], device=device))
        prior_broad = BoxUniform(low=torch.tensor([-15.0, 0.001], device=device), high=torch.tensor([-3.0, 1.0], device=device))

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # NEW: Parallelized Simulation
        # -----------------------------------------------------------------------
        # Automatically determine the number of CPU cores to use for parallelization.
        # We leave one core free to ensure system stability.
        try:
            # os.cpu_count() is more reliable
            n_workers = max(1, os.cpu_count() - 1)
        except NotImplementedError:
            # Fallback for some systems
            n_workers = max(1, torch.multiprocessing.cpu_count() - 1)

        tqdm.write(f"   🧠 Training Rep {r+1} with Focused Sampling...")
        tqdm.write(f"      - 🚀 Parallelizing simulation across {n_workers} workers.")
        tqdm.write(f"      - Simulating {n_weak_sims} samples from weak-signal prior [-10, -5]")
        
        # Call simulate_for_sbi with the num_workers argument
        theta_weak, x_weak = simulate_for_sbi(sim, proposal=prior_weak, num_simulations=n_weak_sims, num_workers=n_workers)
        
        tqdm.write(f"      - Simulating {n_broad_sims} samples from broad prior [-15, -3]")
        theta_broad, x_broad = simulate_for_sbi(sim, proposal=prior_broad, num_simulations=n_broad_sims, num_workers=n_workers)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # --- Combine, Shuffle, and Train (identical) ---
        theta = torch.cat((theta_weak, theta_broad), dim=0)
        x = torch.cat((x_weak, x_broad), dim=0)
        permutation = torch.randperm(theta.size(0))
        theta = theta[permutation]
        x = x[permutation]
        
        inf_ai = SNPE(prior=prior_broad, density_estimator="maf", device=device)
        inf_ai.append_simulations(theta, x)
        post_ai = inf_ai.build_posterior(inf_ai.train(show_train_summary=False))
        
        inf_tr = SNPE(prior=prior_broad, density_estimator="maf", device=device)
        inf_tr.append_simulations(theta, x[:, [0, 1]])
        post_tr = inf_tr.build_posterior(inf_tr.train(show_train_summary=False))

        # --- Save & Eval (identical) ---
        torch.save(post_ai, os.path.join(model_save_dir, f"model_ai_rep{r}.pt"))
        torch.save(post_tr, os.path.join(model_save_dir, f"model_trad_rep{r}.pt"))
        eval_pbar = tqdm(xi_values, desc=f"   📉 Evaluating Rep {r+1}", leave=False)
        for j, xi in enumerate(eval_pbar):
            res_ai[r, j] = calculate_limit(post_ai, sim, xi, "ai", n_trials=20)
            res_tr[r, j] = calculate_limit(post_tr, sim, xi, "trad", n_trials=20)

        global_pbar.update(1)

    FINAL_RESULTS[label] = {"ai": res_ai, "trad": res_tr, "xi": xi_values}
    torch.save(FINAL_RESULTS, SAVE_FILE)
    tqdm.write(f"✅ {label} Saved!")

global_pbar.close()
print(f"\n🎉 V4.2 (Parallel & Focused) Training Complete! Results saved to: {SAVE_FILE}")
