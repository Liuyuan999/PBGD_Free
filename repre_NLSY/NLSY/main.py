import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import load_nlsy79
from model import Fea, Reg, FeaReg
# from utils import train_implicit, evaluate_pp_implicit, 
from utils import train_implicit, train_PBGD_Free, train_PBGD, evaluate_pp_implicit, evaluate_pp_model



import warnings
# Suppress the specific UserWarning about target size mismatch
warnings.filterwarnings("ignore", message=".*Using a target size.*")



X_train, X_test, A_train, A_test, y_train, y_test = load_nlsy79()

# Number of runs for each method
num_runs = 10
num_iter = 10
methods = {
    "train_PBGD": lambda *args: train_PBGD(*args, gam=10),
    "train_PBGD_Free": lambda fea, reg_0, reg_1, criterion, optim_fea, optim_reg_0, optim_reg_1, X_train, A_train, y_train: train_PBGD_Free(fea, reg_0, criterion, optim_fea, optim_reg_0, X_train, A_train, y_train, gam=10),
    "train_implicit": lambda *args: train_implicit(*args, kappa=2e-3)
}

results = {}

for method_name, train_method in methods.items():
    ap_tests = []
    times = []
    iteration_mse = []
    
    
    for i in range(num_runs):
        # Initialize models with the same seed for reproducibility
        torch.manual_seed(i)
        fea = Fea(input_size=len(X_train[0])).cuda()
        reg_0 = Reg().cuda()
        reg_1 = Reg().cuda()

        optim_fea = optim.Adam(fea.parameters(), lr=1e-3, eps=1e-3)
        optim_reg_0 = optim.Adam(reg_0.parameters(), lr=1e-3, eps=1e-3)
        optim_reg_1 = optim.Adam(reg_1.parameters(), lr=1e-3, eps=1e-3)
        criterion = nn.MSELoss()

        # Evaluate at the begining of each iteration
        ap_test_iteration = []  # Record MSE for each iteration
        time_per_iteration = [0]  # Time taken for each iteration
        fea_reg = FeaReg(fea, reg_0)
        ap_test, _, _, _ = evaluate_pp_model(fea_reg, X_test, y_test, A_test)
        ap_test_iteration.append(ap_test)

        # Train model and measure time
        start_time = time.time()
        for iter in tqdm(range(num_iter), desc=f"{method_name} Run {_+1}"):
            iter_start_time = time.time()

            if method_name == "train_PBGD_Free":
                train_method(fea, reg_0, reg_1, criterion, optim_fea, optim_reg_0, optim_reg_1, X_train, A_train, y_train)
            else:
                train_method(fea, reg_0, reg_1, criterion, optim_fea, optim_reg_0, optim_reg_1, X_train, A_train, y_train)

            iter_end_time = time.time()
            time_per_iteration.append(time_per_iteration[-1]+ iter_end_time - iter_start_time)

            # Evaluate at the end of each iteration (excluding evaluation time from training time)
            fea_reg = FeaReg(fea, reg_0)
            ap_test, _, _, _ = evaluate_pp_model(fea_reg, X_test, y_test, A_test)
            ap_test_iteration.append(ap_test)

        end_time = time.time()

        # Record mean and std for the evaluation
        ap_tests.append(ap_test_iteration)
        times.append(time_per_iteration)  # Total time excluding evaluation

    ap_tests = np.array(ap_tests)
    times = np.array(times)
    # Store results for plotting
    results[method_name] = {
        "mean_ap_test": np.mean(ap_tests, axis=0)[-1],
        "std_ap_test": np.std(ap_tests, axis=0)[-1],
        "mean_time": np.mean(times, axis=0)[-1],
        "std_time": np.std(times, axis=0)[-1],
        "iteration_mse": np.mean(ap_tests, axis=0),
        "iteration_mse_sd": np.std(ap_tests, axis=0),
        "iteration_time": np.mean(times, axis=0),
        "iteration_time_sd": np.std(times, axis=0)
    }

# Plot MSE vs Time
plt.figure(figsize=(10, 5))
for method, stats in results.items():
    label_name = str(method)[5:]
    plt.plot(stats["iteration_time"], stats["iteration_mse"], label=label_name)
    fill_up = stats["iteration_mse"] + stats["iteration_mse_sd"]
    fill_down = stats["iteration_mse"] -stats["iteration_mse_sd"]
    plt.fill_between(stats["iteration_time"][1:],fill_up[1:],fill_down[1:],alpha=0.2)
plt.xlabel("Time (seconds)")
plt.ylabel("MSE")
plt.title("MSE vs Time")
plt.legend()
plt.grid("--")
plt.savefig("plots/mse_vs_time.pdf")  # Save the plot as a PDF
plt.close()

# Plot MSE vs Iteration Count
plt.figure(figsize=(10, 5))
for method, stats in results.items():
    label_name = str(method)[5:]
    plt.plot(np.arange(num_iter+1), stats["iteration_mse"], label=label_name)
    fill_up = stats["iteration_mse"] + stats["iteration_mse_sd"]
    fill_down = stats["iteration_mse"] -stats["iteration_mse_sd"]
    plt.fill_between(np.arange(num_iter+1)[1:],fill_up[1:],fill_down[1:],alpha=0.2)
plt.xlabel("Iteration Count")
plt.ylabel("MSE")
plt.title("MSE vs Iteration Count")
plt.legend()
plt.grid("--")
plt.savefig("plots/mse_vs_iteration_count.pdf")  # Save the plot as a PDF
plt.close()

# Print results
for method, stats in results.items():
    print(f"Results for {method}:")
    print(f"  Mean MSE: {stats['mean_ap_test']:.4f} ± {stats['std_ap_test']:.4f}")
    print(f"  Mean Time: {stats['mean_time']:.2f} ± {stats['std_time']:.2f} seconds\n")