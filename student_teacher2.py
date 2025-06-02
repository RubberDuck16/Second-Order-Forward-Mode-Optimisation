import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax import jacobian
from jax.flatten_util import ravel_pytree
from Utils.functions import *
from nn_functions import *


key = random.PRNGKey(465)
teacher_key, student_key = random.split(key, 2)

# Network parameters
no_of_samples = 1000      # no of training samples
di = 11                   # input dimension
Nh = 21                   # number of hidden neurons
do = 6                    # output dimension

P = Nh * (di + 1) + do * Nh
print('P:', P)

teacher_params = random_params(di, Nh, do, teacher_key)     # initialise ground truth network parameters


## ------- PART 1: Generate data using a shallow neural network ---------- ##

# Generate covariance matrix of input distribution 
alpha = 1.5
eigenvalues = np.array([1 / (i + 1)**alpha for i in range(di)])  # power law

# Exponential eigenvalue decay
beta = 0.99
eigenvalues = np.asarray([np.exp(-beta * k) for k in range(di)])

Q, _ = np.linalg.qr(np.random.randn(di, di))
sigma = Q @ np.diag(eigenvalues) @ Q.T              # covariance matrix

# Sample input data
samples_key = random.PRNGKey(13)
z = random.normal(samples_key, (di, no_of_samples))         # input vector (di x no_of_samples)
x = np.linalg.cholesky(sigma) @ z

y = nn_output(teacher_params, x)       # output vector (do x no_of_samples) 



## ------- PART 2: Kronecker Product Approximation of the ground truth GGN ---------- ##

flat_params, unravel_fn = ravel_pytree(teacher_params)

J = jacobian(nn_output_flat)(flat_params, unravel_fn, x)       # shape: (do, no_of_samples, P)
J = J.reshape(-1, flat_params.shape[0])                # shape: (do * no_of_samples, P)

# Hessian of the loss function (second derivative of the loss function w.r.t. the outputs) is 2/no_of_samples * I
G = 2/no_of_samples * (J.T @ J)

layers = [(Nh, di+1), (do, Nh)]            
learned_G, losses, losses1, losses2 = learn_G3(layers, G, iters=20000, K=10)

# get the eigenvectors of the learned GGN
eigenvecs, eigenvalues = get_eigenvectors(learned_G)

# plot actual eigenspectrum
_, s, _ = np.linalg.svd(G)
plt.plot(s**2, label='ground truth GGN eigenvalues')
plt.plot(eigenvalues, label='approximated GGN eigenvalues')
plt.legend()
plt.yscale('log')
plt.show()



## ------- PART 3: Training the parameters of the student network ---------- ##

student_params = random_params(di, Nh, do, student_key)

K = 4          
N = 2000         # number of iterations

losses = nn_SOFO_eigs(K, N, student_params, x, y, eigenvecs, learning_rate=1)
plt.plot(losses, label='SOFO_eigs')

losses = nn_SOFO(K, N, student_params, x, y, learning_rate=1)
plt.plot(losses, label='SOFO')

losses = gradient_descent(N, student_params, x, y, learning_rate=1)
plt.plot(losses, label='Gradient Descent')

plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Total Loss')
plt.yscale('log')
plt.legend()
plt.show()