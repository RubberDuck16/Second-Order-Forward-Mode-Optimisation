import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax import grad, value_and_grad
from jax import jacobian, hessian
from jax.flatten_util import ravel_pytree
from Utils.functions import *


## PART 1: Generate data using a shallow neural network ##

no_of_samples = 600      # no of training samples
di = 5                   # input dimension
Nh = 17                  # number of hidden neurons
do = 3                   # output dimension

P = Nh * (di + 1) + do * Nh
print('P:', P)


## Generate covariance matrix from which we will sample inputs ##
alpha = 1.5
eigenvalues = np.array([1 / (i + 1)**alpha for i in range(di)])  # power law
Q, _ = np.linalg.qr(np.random.randn(di, di))
sigma = Q @ np.diag(eigenvalues) @ Q.T



def random_params(di, Nh, do, key):
    w_key, c_key = random.split(key)
    C = 1/jnp.sqrt((di+1)) * random.normal(c_key, (Nh, di + 1)) 
    W = 1/jnp.sqrt(Nh) * random.normal(w_key, (do, Nh)) 
    return {"C": C, "W": W}



def shallow_nn(x, params):
    x = jnp.vstack((x, jnp.ones((1, x.shape[1]))))           # Add 1 as the bias input
    h = jnp.tanh(jnp.matmul(params["C"], x)) 
    y = jnp.matmul(params["W"], h)  
    return y                                          

def output_from_flat_params(flat_params, unravel_fn, x):
    params = unravel_fn(flat_params)
    return shallow_nn(x, params)        # shape: (do, no_of_samples)


# Sample input data
samples_key = random.PRNGKey(13)
z = random.normal(samples_key, (di, no_of_samples))         # input vector (di x no_of_samples)
x = np.linalg.cholesky(sigma) @ z


# Initialise parameters of teacher network
teacher_key = random.PRNGKey(2)
teacher_params = random_params(di, Nh, do, teacher_key)

# Get the output of the network
y = shallow_nn(x, teacher_params)           # output vector (do x no_of_samples) 
# now have x and y as training data



## Compute the ground truth GGN of the ground truth network ##

flat_teacher_params, teacher_unravel_fn = ravel_pytree(teacher_params)

# Jacobian of the ground truth network parameters
J = jacobian(output_from_flat_params)(flat_teacher_params, teacher_unravel_fn, x)       # shape: (do, no_of_samples, P)
J = J.reshape(-1, flat_teacher_params.shape[0])                                         # shape: (do * no_of_samples, P)

# Hessian of the loss function (second derivative of the loss function w.r.t. the outputs) is 2/no_of_samples * I
GGN = 2/no_of_samples * (J.T @ J)


layers = [(Nh, di+1), (do, Nh)]            

learned_G, losses, losses1, losses2 = learn_G3(layers, GGN, iters=25000, K=10)

plt.plot(losses1)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Layer 1')
plt.yscale('log')
plt.show()

plt.plot(losses2)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Layer 2')
plt.yscale('log')
plt.show()

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Total Loss')
plt.show()