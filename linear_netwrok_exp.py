import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import jacobian, jvp, vmap, value_and_grad, jit
from jax.flatten_util import ravel_pytree
from MAIN_ALGORITHMS import *
import optax

## Network set-up
def random_params(di, do, key):
    W = 1/jnp.sqrt(di) * random.normal(key, (do, di)) 
    return {"W": W}

def linear_network(x, params):
    y = jnp.matmul(params["W"], x)  # shape: (do, no_of_samples)
    return y
                            
def output_from_flat_params(flat_params, unravel_fn, x):
    params = unravel_fn(flat_params)
    return linear_network(x, params)        # shape: (do, no_of_samples)

def MSE_loss(flat_params, unravel_fn, x, y):
    y_pred = output_from_flat_params(flat_params, unravel_fn, x)
    return jnp.mean((y - y_pred) ** 2)

def Jv_fn(flat_params, unravel_fn, x, V):
    def model_fn(p):
        return output_from_flat_params(p, unravel_fn, x).ravel()

    # jvp for a single tangent vector
    def single_jvp(v_col):
        _, jv = jvp(model_fn, (flat_params,), (v_col,))   # jv is (d_o * N,)
        return jv

    # vmap across columns of V  (axis 1)
    Jv = vmap(single_jvp, in_axes=1, out_axes=1)(V)       # shape (d_o * N, K)
    return Jv


def nn_SOFO(K, N, initial_params, x, y, learning_rate=1):
    """
    K = sketching dimension
    N = number of iterations
    initial_params = initial parameters in the form of a pytree
    x = input data
    y = labels for input data
    learning_rate = step size
    """
    losses = []
    
    current_theta, unravel_fn = ravel_pytree(initial_params)    # flatten the parameters into a 1D array
    P = current_theta.shape[0]
    
    no_of_samples = x.shape[1]

    grad_loss_fn = value_and_grad(MSE_loss, argnums=0)

    for n in range(N):  
        # sketching matrix (random tangents)
        V = np.random.randn(P, K)  

        loss, gradient = grad_loss_fn(current_theta, unravel_fn, x, y)      # gradient of cost function
        losses.append(loss)

        # use jvp to sketch: compute J @ V 
        Jv = Jv_fn(current_theta, unravel_fn, x, V)  # shape: (N * do, K)

        # compute GGN
        G_sketched = (2 / no_of_samples) * (Jv.T @ Jv)  # K x K
        
        g = V.T @ gradient          # sketched gradient (K x 1)

        # parameter update
        dtheta = V @ (jnp.linalg.solve(G_sketched, g))
        current_theta = current_theta - learning_rate * dtheta

        if n % 100 == 0:
            print(f"Step {n} | Loss: {loss:.6f}")
    return losses


def nn_SOFO_eigs(K, N, initial_params, x, y, approx_G, learning_rate=1):
    """
    K = sketching dimension
    N = number of iterations
    initial_params = initial parameters in the form of a pytree
    x = input data
    y = labels for input data
    approx_G = the approximated GGN
    learning_rate = step size
    """
    losses = []

    if isinstance(approx_G, list):
        U, _ = get_eigenvectors(approx_G)   
    else:
        U, _, _ = np.linalg.svd(approx_G)
    
    current_theta, unravel_fn = ravel_pytree(initial_params)    # flatten the parameters into a 1D array
    P = current_theta.shape[0]
    
    no_of_samples = x.shape[1]

    grad_loss_fn = value_and_grad(MSE_loss, argnums=0)

    for n in range(N):  
        # sketching matrix (eigenvector tangents)
        indices = (np.arange(K) + (K*n % P)) % P
        V = U[:, indices]     

        loss, gradient = grad_loss_fn(current_theta, unravel_fn, x, y)      # gradient of cost function
        losses.append(loss)

        # use jvp to sketch: compute J @ V 
        Jv = Jv_fn(current_theta, unravel_fn, x, V)  # shape: (N * do, K)

        # compute GGN
        G_sketched = (2 / no_of_samples) * (Jv.T @ Jv)  # K x K
        
        g = V.T @ gradient          # sketched gradient of cost function (K x 1)

        # parameter update
        dtheta = V @ (jnp.linalg.solve(G_sketched, g))
        current_theta = current_theta - learning_rate * dtheta

        if n % 100 == 0:
            print(f"Step {n} | Loss: {loss:.6f}")
    return losses


def adam_training(N, initial_params, x, y, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Train for N iterations with Adam.

    Parameters
    ----------
    N              : int                – number of iterations
    initial_params : pytree            – initial parameters
    x              : ndarray           – input data
    y              : ndarray           – target labels
    learning_rate  : float             – Adam lr (default 1e-3)
    beta1, beta2   : float             – Adam momentum coefficients
    eps            : float             – Adam epsilon
    """

    # Flatten parameters to 1-D vector
    theta0, unravel_fn = ravel_pytree(initial_params)

    # Optax optimiser
    opt = optax.adam(learning_rate, b1=beta1, b2=beta2, eps=eps)
    opt_state = opt.init(theta0)

    # Value-and-grad function on the flat parameter vector
    loss_and_grad = value_and_grad(MSE_loss, argnums=0)

    losses = []

    @jit
    def update_step(theta, opt_state):
        loss, g = loss_and_grad(theta, unravel_fn, x, y)
        updates, opt_state2 = opt.update(g, opt_state)
        theta2 = optax.apply_updates(theta, updates)
        return theta2, opt_state2, loss

    theta = theta0
    for n in range(N):
        theta, opt_state, loss_val = update_step(theta, opt_state)
        losses.append(loss_val)

        if n % 100 == 0:
            print(f"Step {n:5d} | Loss: {loss_val:.6f}")

    return losses




   
di = 27                  # input dimension
do = 13                  # output dimension

P = di * do
print('P:', P)

layers = [(do, di)]    

no_of_samples = 800  

K = 3
no_of_iters = 800

key = random.PRNGKey(0)
key1, key2, key3 = random.split(key, 3)


alpha = 1.5
eigenvalues = np.array([1 / (i + 1)**alpha for i in range(di)])  # power law
Q, _ = np.linalg.qr(np.random.randn(di, di))
sigma = Q @ np.diag(eigenvalues) @ Q.T

# Sample input data
z = random.normal(key1, (di, no_of_samples))         # input vector (di x no_of_samples)
x = np.linalg.cholesky(sigma) @ z

# Ground truth network parameters
teacher_params = random_params(di, do, key2)
flat_teacher_params, teacher_unravel_fn = ravel_pytree(teacher_params)

y = linear_network(x, teacher_params)           # output vector (do x no_of_samples) 

print('teacher_params shape:', teacher_params["W"].shape)
print('x shape:', x.shape)
print('y shape:', y.shape)
print('No. of parameters:', len(ravel_pytree(teacher_params)[0]))

J = jacobian(output_from_flat_params)(flat_teacher_params, teacher_unravel_fn, x)       # shape: (do, no_of_samples, P)
J = J.reshape(-1, flat_teacher_params.shape[0])                                         # shape: (do * no_of_samples, P)
GGN = 2/no_of_samples * (J.T @ J)

learned_G, losses = learn_G_one_layer(layers, GGN, iters=20000, K=10)

plt.plot(losses)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Log Auxiliary Loss', fontsize=15)
plt.yscale('log')
plt.savefig('learned_G_loss_linearnetwork_final.png', dpi=300)
plt.show()

initial_params = random_params(di, do, key3)

SOFO_losses = nn_SOFO(K, no_of_iters, initial_params, x, y, learning_rate=10)
eig_SOFO_losses = nn_SOFO_eigs(K, no_of_iters, initial_params, x, y, learned_G, learning_rate=10)
adam_losses = adam_training(no_of_iters, initial_params, x, y, learning_rate=1e-3)

plt.plot(SOFO_losses, label='SOFO')
plt.plot(eig_SOFO_losses, label='EIG-SOFO')
plt.plot(adam_losses, label='Adam')


no_of_samples = 800  

K = 3
no_of_iters = 800

key = random.PRNGKey(0)
key1, key2, key3 = random.split(key, 3)


alpha = 1.991
eigenvalues = np.array([1 / (i + 1)**alpha for i in range(di)])  # power law
Q, _ = np.linalg.qr(np.random.randn(di, di))
sigma = Q @ np.diag(eigenvalues) @ Q.T

# Sample input data
z = random.normal(key1, (di, no_of_samples))         # input vector (di x no_of_samples)
x = np.linalg.cholesky(sigma) @ z

# Ground truth network parameters
teacher_params = random_params(di, do, key2)
flat_teacher_params, teacher_unravel_fn = ravel_pytree(teacher_params)

y = linear_network(x, teacher_params)           # output vector (do x no_of_samples) 

print('teacher_params shape:', teacher_params["W"].shape)
print('x shape:', x.shape)
print('y shape:', y.shape)
print('No. of parameters:', len(ravel_pytree(teacher_params)[0]))

J = jacobian(output_from_flat_params)(flat_teacher_params, teacher_unravel_fn, x)       # shape: (do, no_of_samples, P)
J = J.reshape(-1, flat_teacher_params.shape[0])                                         # shape: (do * no_of_samples, P)
GGN = 2/no_of_samples * (J.T @ J)

learned_G, losses = learn_G_one_layer(layers, GGN, iters=25000, K=10)

initial_params = random_params(di, do, key3)

SOFO_losses = nn_SOFO(K, no_of_iters, initial_params, x, y, learning_rate=10)
eig_SOFO_losses = nn_SOFO_eigs(K, no_of_iters, initial_params, x, y, learned_G, learning_rate=10)
adam_losses = adam_training(no_of_iters, initial_params, x, y, learning_rate=1e-3)

plt.plot(SOFO_losses, label='SOFO', linestyle='--')
plt.plot(eig_SOFO_losses, label='EIG-SOFO', linestyle='--')
plt.plot(adam_losses, label='Adam', linestyle='--')

plt.title(f'Large Data Regime ({no_of_samples} samples)', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Log Loss', fontsize=14)
plt.yscale('log')
plt.legend(fontsize=12)
plt.show()
