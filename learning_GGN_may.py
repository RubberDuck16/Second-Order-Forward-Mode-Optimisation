import matplotlib.pyplot as plt
from Utils.functions import *
from Utils.algorithms import *
from scipy.linalg import block_diag


### MAKE BLOCKS ###
layers = [(7, 4), (7, 4)]

P = 0
for n_left, n_right in layers:
    P += n_left*n_right


# block 1
beta = 0.9
n_left, n_right = layers[0]
eigs_left = [np.exp(-beta * k) for k in range(n_left)]
eigs_right = [np.exp(-beta * k) for k in range(n_right)]
block1 = [initialise_g(n_left, n_right, jax.random.PRNGKey(80), eigs_left=eigs_left, eigs_right=eigs_right)]   # list of KPs describing the block
matrix1 = KP_sum(block1)

# block 2
beta = 0.9  
n_left, n_right = layers[1]
eigs_left = [np.exp(-beta * k) for k in range(n_left)]
eigs_right = [np.exp(-beta * k) for k in range(n_right)]
block2 = [initialise_g(n_left, n_right, jax.random.PRNGKey(80), eigs_left=eigs_left, eigs_right=eigs_right)]   # list of KPs describing the block
matrix2 = KP_sum(block2)

true_GGN = block_diag(matrix1, matrix2)

learned_G, losses, losses1, losses2 = learn_G3(layers, true_GGN, iters=20000, K=10)

plt.plot(losses1, label='Loss1')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.plot(losses2, label='Loss2')
plt.legend()
plt.show()

plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.yscale('log')   
plt.show()


# block 1
alpha = 1.5
n_left, n_right = layers[0]
eigs_left = [1/(i+1)**alpha for i in range(n_left)]
eigs_right = [1/(i+1)**alpha for i in range(n_right)]
block1 = [initialise_g(n_left, n_right, jax.random.PRNGKey(80), eigs_left=eigs_left, eigs_right=eigs_right)]   # list of KPs describing the block
matrix1 = KP_sum(block1)

# block 2
alpha = 1.5
n_left, n_right = layers[1]
eigs_left = [1/(i+1)**alpha for i in range(n_left)]
eigs_right = [1/(i+1)**alpha for i in range(n_right)]
block2 = [initialise_g(n_left, n_right, jax.random.PRNGKey(80), eigs_left=eigs_left, eigs_right=eigs_right)]   # list of KPs describing the block
matrix2 = KP_sum(block2)

true_GGN = block_diag(matrix1, matrix2)

learned_G, losses, losses1, losses2 = learn_G3(layers, true_GGN, iters=20000, K=10)

plt.plot(losses1, label='Loss1')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.plot(losses2, label='Loss2')
plt.legend()
plt.show()

plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.yscale('log')   
plt.show()