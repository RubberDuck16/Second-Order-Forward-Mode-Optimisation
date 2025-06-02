import numpy as np
from jax import random
import matplotlib.pyplot as plt
from Utils.algorithms import *

def GGN(eigenvalues):
    # Return a PxP curvature matrix from a list of P eigenvalues
    P = len(eigenvalues)
    u, _ = np.linalg.qr(np.random.randn(P, P)) 
    s = np.diag(np.array(eigenvalues)) 
    return u @ s @ u.T

def plot_eigenspectra(eigenvalue_options):
    plt.figure(figsize=(10, 6))
    for label, eigenvalues in eigenvalue_options.items():
        P = len(eigenvalues)
        plt.plot(range(25), eigenvalues[:25], marker='o', linestyle='-', label=label)
    plt.title('Eigenspectra', fontsize=18)
    plt.xlabel('Index', fontsize=15)
    plt.ylabel('Eigenvalue', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.show()

## Parameters
P = 200
iterations = 100
N_trials = 50

beta, alpha = 0.05, 1.991

eigenvalue_options = {
    "Uniform": [1 + 0.1 * np.random.uniform(-1, 1) for _ in range(P)],
    "Exponential Decay": [10 * np.exp(-beta * i) for i in range(P)],
    "Dominant Eigenvalue": list(np.array([100] + [1] * (P - 1))),
    "Power Law (β = 1.991)": [1/(i+1)**alpha for i in range(P)]
}

#plot_eigenspectra(eigenvalue_options)

Ks = [2, 20]
colors = ['b', 'm']

fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True) 

indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
j = 0

for label, eigenvalues in eigenvalue_options.items():
    M = random_pos_def_sqrt(P, jax.random.PRNGKey(0), np.array(eigenvalues))
    sigma = M @ M.T
    print(np.linalg.cond(sigma))

    for i, K in enumerate(Ks):
        ratio = K/P
        total_losses1 = np.zeros((N_trials, iterations))
        total_losses2 = np.zeros((N_trials, iterations))
        no_of_steps1 = 0
        no_of_steps2 = 0
        for trial in range(N_trials):
            key1, key2 = jax.random.split(jax.random.PRNGKey(trial))
            M = random_pos_def_sqrt(P, key1, np.array(eigenvalues))
            sigma = M @ M.T
            print('Trial:', trial) 
            total_losses1[trial][:], steps = original_SOFO_loss(P, K, sigma, key2, iterations)
            #total_losses2[trial][:], _ = truncated_CT_SOFO_loss(P, K, sigma, key, iterations, truncation=10)
            total_losses2[trial][:], steps = CT_SOFO_loss(K, sigma, key2, iterations)
            no_of_steps1 += steps
            no_of_steps2 += steps
        
        avg_steps1 = no_of_steps1 / N_trials
        avg_steps2 = no_of_steps2 / N_trials
        print(f'Average steps for SOFO: {avg_steps1}, CT-SOFO: {avg_steps2} for {label} with K={K}')

        median_loss1 = np.median(total_losses1, axis=0)
        median_loss2 = np.median(total_losses2, axis=0)
        per_25_loss1 = np.percentile(total_losses1, 25, axis=0)
        per_25_loss2 = np.percentile(total_losses2, 25, axis=0)
        per_75_loss1 = np.percentile(total_losses1, 75, axis=0)
        per_75_loss2 = np.percentile(total_losses2, 75, axis=0)
     
        axs[indices[j][0]][indices[j][1]].plot(np.linspace(0, iterations, iterations), median_loss1, linestyle='--', color=colors[i], label=f'SOFO, {int(ratio*100)}%')
        axs[indices[j][0]][indices[j][1]].fill_between(np.linspace(0, iterations, iterations), per_25_loss1, per_75_loss1, alpha=0.1, color='gray')
        axs[indices[j][0]][indices[j][1]].plot(np.linspace(0, iterations, iterations), median_loss2, color=colors[i], label=f'CT-SOFO, {int(ratio*100)}%')
        axs[indices[j][0]][indices[j][1]].fill_between(np.linspace(0, iterations, iterations), per_25_loss2, per_75_loss2, alpha=0.1, color='gray')
        axs[indices[j][0]][indices[j][1]].set_yscale('log')
  
    axs[indices[j][0]][indices[j][1]].set_title(f'{label} Eigenspectrum', fontsize=15)
    axs[indices[j][0]][indices[j][1]].set_ylabel('Log Loss', fontsize=15)
    axs[indices[j][0]][indices[j][1]].legend(fontsize=15)
    axs[indices[j][0]][indices[j][1]].set_xlabel('Iteration', fontsize=15)
    j += 1

plt.tight_layout()
plt.subplots_adjust(hspace=0.22) 
plt.savefig(f'Plots/{label}_losses.png')
plt.show()


