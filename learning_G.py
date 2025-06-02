import numpy as np
import matplotlib.pyplot as plt
from Utils.functions import *

"""
This script performs the following tasks:
1. Initializes three different eigenvalue spectrums for left and right matrices.
2. Generates an initial guess for the matrix G which is identity matrix.
3. Iteratively optimizes the approximation of G using a Kronecker product.
4. Plots the eigenvalues of the left matrix, the training loss, and the log training loss.
"""

CASE1 = True        # Case 1: GGN is a KP
CASE2A = False      # Case 2a: GGN is a sum of KP with a dominant KP
CASE2B = False      # Case 2b: GGN is a sum of KP with equally-weighted KP
CASE3 = False       # Case 3: GGN is not a KP

n_left = 27
n_right = 13
K = 5                    
P = n_left * n_right

n = n_left
eigenvalues_l = [
    10 * jnp.exp(-jnp.arange(n) / (n / 4)),
    10 * jnp.exp(-jnp.arange(n) / (n / 8)),
    jnp.array([10 * (0.5 ** i) for i in range(n)])
]

n = n_right
eigenvalues_r = [
    10 * jnp.exp(-jnp.arange(n) / (n / 4)),
    10 * jnp.exp(-jnp.arange(n) / (n / 8)),
    jnp.array([10 * (0.5 ** i) for i in range(n)])
]

if CASE3:
    n = P
    eigenvalues = [
        jnp.exp(-jnp.arange(P) / (P / 4)),
        jnp.exp(-jnp.arange(P) / (P / 8)),
        jnp.array([(0.5 ** i) for i in range(P)])
    ]

# Initial guess of G
initial_guess = [
    identity_guess(n_left, n_right)
]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):

    if CASE1:
        true_G = [initialise_g(n_left, n_right, jax.random.PRNGKey(1), eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i])]
    elif CASE2A:
        true_G = [
            initialise_g(n_left, n_right, jax.random.PRNGKey(1), eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i]),
            initialise_g(n_left, n_right, jax.random.PRNGKey(2), alpha=0.2, eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i]),
            initialise_g(n_left, n_right, jax.random.PRNGKey(8), alpha=0.1, eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i])
        ]
    elif CASE2B:
        true_G = [
            initialise_g(n_left, n_right, jax.random.PRNGKey(1), eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i]),
            initialise_g(n_left, n_right, jax.random.PRNGKey(2), eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i]),
            initialise_g(n_left, n_right, jax.random.PRNGKey(8), eigs_left=eigenvalues_l[i], eigs_right=eigenvalues_r[i])
        ]
    else:
        true_G = ground_truth_G(P, key=jax.random.PRNGKey(1), eigs=eigenvalues[i])

    G_approx, losses = optimise_G_hat(initial_guess, true_G, K, iters=50000)

    axs[0].plot(eigenvalues_l[i], label=f'Left eigenspectrum {i}', alpha=0.7)
    axs[0].set_xlabel('Eigenvalue index')
    axs[0].set_ylabel('Eigenvalue')
    axs[0].legend()
    axs[0].grid(alpha=0.5)
    axs[1].plot(losses, label=f'Loss {i}', alpha=0.7)
    axs[1].set_xlabel('Iteration')
    axs[1].legend()
    axs[1].grid(alpha=0.5)
    axs[2].plot(losses, label=f'Loss {i}', alpha=0.7)
    axs[2].set_xlabel('Iteration')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].legend()
    axs[2].grid(alpha=0.5)

axs[0].set_title('Left Eigenspectrums of Ground Truth G')
axs[1].set_title('Training Loss')
axs[2].set_title('Log Training Loss')

if CASE1:
    fig.suptitle(f'Learning a Kronecker Product Approximation of G which is a KP, P = {P}, K = {K}')
elif CASE2A:
    fig.suptitle(f'Learning a Kronecker Product Approximation of G which is a sum of KP with a dominant KP, P = {P}, K = {K}')
elif CASE2B:    
    fig.suptitle(f'Learning a Kronecker Product Approximation of G which is an equally-weighted sum of KP, P = {P}, K = {K}')
else:
    fig.suptitle(f'Learning a Kronecker Product Approximation of G which is not a KP, P = {P}, K = {K}')

plt.constrained_layout=True
plt.show()






