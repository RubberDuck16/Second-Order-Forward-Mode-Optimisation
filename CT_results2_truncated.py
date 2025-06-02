import numpy as np
from jax import random
import matplotlib.pyplot as plt
from Utils.algorithms import *
from Utils.functions import *

fig, axs = plt.subplots(2, 1, figsize=(7, 15))

## Parameters
P = 200
K = 2
iterations = 100

beta = 0.6
eigs = [np.exp(-beta * i) for i in range(P)]

M = random_pos_def_sqrt(P, jax.random.PRNGKey(5), np.array(eigs))
sigma = M @ M.T

# Second subplot: CT-SOFO vs. SOFO over 10 trials and plot the median
key = jax.random.PRNGKey(7)
num_trials = 10

losses1 = []
losses2 = []
losses3 = []
losses4 = []

for i in range(num_trials):
    subkey = jax.random.split(key, num=2)[0]
    losses1.append(original_SOFO_loss(P, K, sigma, subkey, iterations))
    losses2.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=1))
    losses3.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=5))
    losses4.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=10))
    key = subkey  # Update key for the next trial

# Compute the median across trials
median_losses1 = np.median(losses1, axis=0)
median_losses2 = np.median(losses2, axis=0)
median_losses3 = np.median(losses3, axis=0)
median_losses4 = np.median(losses4, axis=0)

# Plot the median losses
axs[0].plot(np.linspace(0, iterations, iterations), median_losses1, linestyle='--', label=f'SOFO', linewidth=2.5)
axs[0].plot(np.linspace(0, iterations, iterations), median_losses2, label=f'Truncated CT-SOFO (T=1)', linewidth=2.5)
axs[0].plot(np.linspace(0, iterations, iterations), median_losses3, label=f'Truncated CT-SOFO (T=5)', linewidth=2.5)
axs[0].plot(np.linspace(0, iterations, iterations), median_losses4, label=f'Truncated CT-SOFO (T=10)', linewidth=2.5)
axs[0].set_ylabel('Log Loss', fontsize=14)
axs[0].set_title(f'Exponential Decay (α = 0.6)', fontsize=14)
axs[0].legend(fontsize=13)
axs[0].set_yscale('log')




alpha = 1.991
eigs = [1/(i+1)**alpha for i in range(P)]

M = random_pos_def_sqrt(P, jax.random.PRNGKey(5), np.array(eigs))
sigma = M @ M.T



# Second subplot: CT-SOFO vs. SOFO over 10 trials and plot the median
key = jax.random.PRNGKey(7)
num_trials = 10

losses1 = []
losses2 = []
losses3 = []
losses4 = []

for i in range(num_trials):
    subkey = jax.random.split(key, num=2)[0]
    losses1.append(original_SOFO_loss(P, K, sigma, subkey, iterations))
    losses2.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=1))
    losses3.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=5))
    losses4.append(truncated_CT_SOFO_loss(P, K, sigma, subkey, iterations, truncation=10))
    key = subkey  # Update key for the next trial

# Compute the median across trials
median_losses1 = np.median(losses1, axis=0)
median_losses2 = np.median(losses2, axis=0)
median_losses3 = np.median(losses3, axis=0)
median_losses4 = np.median(losses4, axis=0)

# Plot the median losses
axs[1].plot(np.linspace(0, iterations, iterations), median_losses1, linestyle='--', label=f'SOFO', linewidth=2.5)
axs[1].plot(np.linspace(0, iterations, iterations), median_losses2, label=f'Truncated CT-SOFO (T=1)', linewidth=2.5)
axs[1].plot(np.linspace(0, iterations, iterations), median_losses3, label=f'Truncated CT-SOFO (T=5)', linewidth=2.5)
axs[1].plot(np.linspace(0, iterations, iterations), median_losses4, label=f'Truncated CT-SOFO (T=10)', linewidth=2.5)
axs[1].set_xlabel('Iteration', fontsize=14)
axs[1].set_ylabel('Log Loss', fontsize=14)
axs[1].set_title(f'Power Law (β = 1.991)', fontsize=14)
axs[1].set_yscale('log')

fig.suptitle(f'Median Training Loss across {num_trials} Trials, K/P = {int(K/P*100)}%', fontsize=16)
plt.show()
