import numpy as np
from jax import random
import matplotlib.pyplot as plt
from Utils.algorithms import *
from Utils.functions import *

def GGN(eigenvalues):
    # Return a PxP curvature matrix from a list of P eigenvalues
    P = len(eigenvalues)
    u, _ = np.linalg.qr(np.random.randn(P, P)) 
    s = np.diag(np.array(eigenvalues)) 
    return u @ s @ u.T

## Parameters
K = 1
iterations = 500

n_left = 27
n_right = 13
P = n_left * n_right

alpha = 0.05             # % of P that u want to have useful eigenvalues
    
# Choose eigenspectrum
eigs_left = [1/(1+i/(alpha*n_left)) for i in range(n_left)]
eigs_right = [1/(1+i/(alpha*n_right)) for i in range(n_right)]

G = [initialise_g(n_left, n_right, jax.random.PRNGKey(40), eigs_left=eigs_left, eigs_right=eigs_right)]
sigma = KP_sum(G)

eigenvalues, _ = jnp.linalg.eigh(sigma)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

print(np.linalg.cond(sigma))

# First subplot: Eigenvalue Spectrum
axes[0].plot(eigenvalues[::-1], linestyle='-', color='black')
axes[0].set_title(f'Eigenspectrum of Curvature Matrix', fontsize=14)
axes[0].set_ylabel('Eigenvalue (log-scale)', fontsize=14)
axes[0].set_yscale('log')
axes[0].set_xlabel('Index', fontsize=14)
axes[0].grid(alpha=0.5)

# Second subplot: CT-SOFO vs. SOFO
key = jax.random.PRNGKey(7)
total_losses1, _ = original_SOFO_loss(P, K, sigma, key, iterations)
total_losses2, _ = CT_SOFO_loss(K, sigma, key, iterations)
total_losses3, _ = CT_SOFO_loss(2, sigma, key, iterations)

axes[1].plot(np.linspace(0, iterations, iterations), total_losses1, linestyle='--', label=f'SOFO', linewidth=2.5)
axes[1].plot(np.linspace(0, iterations, iterations), total_losses2, label=f'CT-SOFO (K=1)', linewidth=2.5)
axes[1].plot(np.linspace(0, iterations, iterations), total_losses3, label=f'CT-SOFO (K=2)', linewidth=2.5, alpha=0.8)
axes[1].axvline(x=P, color='k', linestyle='--', alpha=0.5)
axes[1].set_title(f'CT-SOFO vs. SOFO (P = {P})', fontsize=14)
axes[1].set_ylabel('Loss', fontsize=14)
axes[1].legend(fontsize=14)
axes[1].set_xlabel('Iteration', fontsize=14)

plt.tight_layout()
plt.show()
