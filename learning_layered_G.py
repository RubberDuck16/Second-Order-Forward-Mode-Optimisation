import matplotlib.pyplot as plt
from Utils.functions import *
from Utils.algorithms import *
from scipy.linalg import block_diag

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

### MAKE BLOCKS ###
layers = [(27, 13), (19, 7)]

P = 0
for n_left, n_right in layers:
    P += n_left*n_right

fig.suptitle(f'{len(layers)} Layers with P = {P} Parameters', fontsize=16)

# block 1
alpha = 0.05   
n_left, n_right = layers[0]
eigs_left = [1/(1+i/(alpha*n_left)) for i in range(n_left)]
eigs_right = [1/(1+i/(alpha*n_right)) for i in range(n_right)]
block1 = [initialise_g(n_left, n_right, jax.random.PRNGKey(4), eigs_left=eigs_left, eigs_right=eigs_right),  
          initialise_g(n_left, n_right, jax.random.PRNGKey(7), eigs_left=eigs_left, eigs_right=eigs_right, alpha=0.3)]
matrix1 = KP_sum(block1)
eigenvalues, _ = jnp.linalg.eigh(matrix1)
axs[0][0].plot(eigenvalues[::-1], linestyle='-', color='black')
axs[0][0].set_title('Eigenvalue Spectrum of Ground Truth G - Layer 1')
axs[0][0].set_yscale('log')
axs[0][0].set_xlabel('Eigenvalue Index')
axs[0][0].grid(alpha=0.5)

# block 2
alpha = 0.2  
n_left, n_right = layers[1] 
eigs_left = [1/(1+i/(alpha*n_left)) for i in range(n_left)]
eigs_right = [1/(1+i/(alpha*n_right)) for i in range(n_right)]
block2 = [initialise_g(n_left, n_right, jax.random.PRNGKey(5), eigs_left=eigs_left, eigs_right=eigs_right),
          initialise_g(n_left, n_right, jax.random.PRNGKey(8), eigs_left=eigs_left, eigs_right=eigs_right, alpha=0.2)]
matrix2 = KP_sum(block2)
eigenvalues, _ = jnp.linalg.eigh(matrix2)
axs[0][1].plot(eigenvalues[::-1], linestyle='-', color='black')
axs[0][1].set_title('Eigenvalue Spectrum of Ground Truth G - Layer 2')
axs[0][1].set_yscale('log')
axs[0][1].set_xlabel('Eigenvalue Index')
axs[0][1].grid(alpha=0.5)

G_list = [block1, block2]
G_matrix = block_diag(matrix1, matrix2)

print(block1[0]["left"] @ block1[0]["left"].T)

# learn G to different points
params1, losses1 = learn_G(layers, G_matrix, iters=500, K=5)
params2, losses2 = learn_G(layers, G_matrix, iters=1000, K=5)
params3, losses3 = learn_G(layers, G_matrix, iters=8000, K=5)

axs[1][0].plot(losses3)
axs[1][0].set_title(f'Loss of Learning G with Sketching Dimension {5}')
axs[1][0].set_yscale('log')
axs[1][0].set_xlabel('Iteration')
axs[1][0].set_ylabel('Loss')
axs[1][0].axvline(x=500, color='purple', linestyle='--')
axs[1][0].axvline(x=1000, color='green', linestyle='--')
axs[1][0].axvline(x=8000, color='red', linestyle='--')
axs[1][0].grid(alpha=0.5)

# training with different algorithms
key = jax.random.PRNGKey(18)
N = 1000
K = 20

sofo_losses = original_SOFO_loss(P, K, G_matrix, key, N)            # SOFO
KP_losses = KP_conj_tangents(layers, K, G_list, key, N, G_list)     # exact KP
KP_losses_est1 = KP_conj_tangents(layers, K, G_list, key, N, params1)
KP_losses_est2 = KP_conj_tangents(layers, K, G_list, key, N, params2)
KP_losses_est3 = KP_conj_tangents(layers, K, G_list, key, N, params3)

axs[1][1].plot(sofo_losses, label="SOFO")
axs[1][1].plot(KP_losses, label="KP (Exact)")
axs[1][1].plot(KP_losses_est1, label="KP (Estimated from 500 iters)", color='purple')
axs[1][1].plot(KP_losses_est2, label="KP (Estimated from 1000 iters)", color='green')
axs[1][1].plot(KP_losses_est3, label="KP (Estimated from 8000 iters)", color='red')
axs[1][1].legend()
axs[1][1].set_yscale("log")
axs[1][1].set_xlabel("Iteration")
axs[1][1].set_ylabel("Log Loss")
axs[1][1].set_title(f"Optimisation of Quadratic Loss, K: {K}")
axs[1][1].grid(alpha=0.5)
plt.show()