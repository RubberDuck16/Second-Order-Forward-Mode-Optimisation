import numpy as np
import matplotlib.pyplot as plt
from jax.nn import relu
import jax.numpy as jnp
from jax import random, jit, jacobian, value_and_grad, vmap, jvp
from jax.flatten_util import ravel_pytree
from helper_functions import *
import optax
from scipy.linalg import block_diag
import pickle

def random_params(di, Nh, do, key):
    w_key, c_key = random.split(key)
    C = 1/jnp.sqrt((di+1)) * random.normal(c_key, (Nh, di + 1)) 
    W = 1/jnp.sqrt(Nh) * random.normal(w_key, (do, Nh)) 
    return {"C": C, "W": W}

def shallow_nn(x, params):
    x = jnp.vstack((x, jnp.ones((1, x.shape[1]))))           # Add 1 as the bias input
    h = jnp.tanh(jnp.matmul(params["C"], x)) 
    #h = relu(jnp.matmul(params["C"], x))
    y = jnp.matmul(params["W"], h)  
    return y                                          

def output_from_flat_params(flat_params, unravel_fn, x):
    params = unravel_fn(flat_params)
    return shallow_nn(x, params)        # shape: (do, no_of_samples)

def compute_GGN(current_flat_params, unravel_fn, x):
    no_of_samples = x.shape[1]
    J = jacobian(output_from_flat_params)(current_flat_params, unravel_fn, x)       # shape: (do, no_of_samples, P)
    J = J.reshape(-1, current_flat_params.shape[0])                                 # shape: (do * no_of_samples, P)
    GGN = 2/no_of_samples * (J.T @ J)
    return GGN

def mse_loss(flat_params, unravel_fn, x, y_true):
    y_pred = output_from_flat_params(flat_params, unravel_fn, x) 
    return jnp.mean(jnp.sum((y_pred - y_true) ** 2, axis=0))  # sum over outputs, mean over samples

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

def nn_SOFO(K, N, initial_params, x, y, learning_rate=1, damping=False, alpha=1e-4):
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

    grad_loss_fn = value_and_grad(mse_loss, argnums=0)

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
        if damping:
            U_damp, s, Vt = np.linalg.svd(G_sketched)
            damping_factor = alpha * np.max(s)
            inverted_G = U_damp * (1 / (s + damping_factor)) @ Vt
            dtheta = V @ inverted_G @ g
        else:
            dtheta = V @ (np.linalg.solve(G_sketched, g))
        current_theta = current_theta - learning_rate * dtheta

        if n % 100 == 0:
            print(f"Step {n} | Loss: {loss:.6f}")
    return losses


def nn_SOFO_eigs2(K, N, initial_params, x, y, layers, approx_freq=500, sketching_iters=25000, approx_K=10, learning_rate=1, damping=False, alpha=1e-4):
    """
    K = sketching dimension
    N = number of iterations
    initial_params = initial parameters in the form of a pytree
    x = input data
    y = labels for input data
    learning_rate = step size
    approx_freq = frequency of approximating the GGN (re-estimate GGN every approx_freq steps)
    sketching_iters = number of iterations for learning the GGN
    approx_K = sketching dimension for learning the GGN
    """
    losses = []
    
    current_theta, unravel_fn = ravel_pytree(initial_params)    # flatten the parameters into a 1D array
    P = current_theta.shape[0]
    
    no_of_samples = x.shape[1]

    grad_loss_fn = value_and_grad(mse_loss, argnums=0)

    for n in range(N):  
        # approximate GGN 
        if n % approx_freq == 0:
            print("Recomputing GGN")
            current_GGN = compute_GGN(current_theta, unravel_fn, x)
            learned_G, _, _, _ = learn_G_multiple_layers(layers, current_GGN, iters=sketching_iters, K=approx_K, learning_rate=1e-4)
            U, _ = get_eigenvectors(learned_G)  

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
        if damping:
            U_damp, s, Vt = np.linalg.svd(G_sketched)
            damping_factor = alpha * np.max(s)
            inverted_G = U_damp * (1 / (s + damping_factor)) @ Vt
            dtheta = V @ inverted_G @ g
        else:
            dtheta = V @ (np.linalg.solve(G_sketched, g))
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
    loss_and_grad = value_and_grad(mse_loss, argnums=0)

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


no_of_trials = 10
no_of_samples = 600      # no of training samples
di = 5                   # input dimension
do = 3
Nh = 10

P = Nh * (di + 1) + do * Nh
print('P:', P)

layers = [(Nh, di+1), (do, Nh)]  


for i in range(2, no_of_trials+2):
    print(f"Trial {i+1}/{no_of_trials} -------------------------------------------------")
    key = random.PRNGKey(87+i)
    key1, key2, key3, key4 = random.split(key, 4)

    ## Generate covariance matrix from which we will sample inputs ##
    alpha = 1.5
    eigenvalues = np.array([1 / (i + 1)**alpha for i in range(di)])  
    Q, _ = np.linalg.qr(random.normal(key1, (di, di)))
    sigma = Q @ np.diag(eigenvalues) @ Q.T

    # Sample input data
    z = random.normal(key2, (di, no_of_samples))         
    x = np.linalg.cholesky(sigma) @ z

    # make ground-truth / teacher params
    teacher_params = random_params(di, Nh, do, key3)
    flat_teacher_params, teacher_unravel_fn = ravel_pytree(teacher_params)

    y = shallow_nn(x, teacher_params)  

    GGN_teacher = compute_GGN(flat_teacher_params, teacher_unravel_fn, x)
    print("GGN_teacher shape:", GGN_teacher.shape)

    student_params = random_params(di, Nh, do, key4)
    flat_student_params, student_unravel_fn = ravel_pytree(student_params)

    no_of_iters = 20000
    K = 2

    adam_losses = adam_training(no_of_iters, student_params, x, y, learning_rate=1e-3)
    sofo_losses = nn_SOFO(K, no_of_iters, student_params, x, y, learning_rate=1)
    sofo_eigs_losses = nn_SOFO_eigs2(K, no_of_iters, student_params, x, y, layers, approx_freq=1000000)
    sofo_eigs_losses2 = nn_SOFO_eigs2(K, no_of_iters, student_params, x, y, layers, approx_freq=500, sketching_iters=25000, approx_K=10)
   
    with open(f'small2_sofo_losses{i}.pkl', 'wb') as f:
        pickle.dump(sofo_losses, f)

    with open(f'small2_sofo_eigs_losses{i}.pkl', 'wb') as f:
        pickle.dump(sofo_eigs_losses, f)

    with open(f'small2_sofo_eigs_losses_keep_learning{i}.pkl', 'wb') as f:
        pickle.dump(sofo_eigs_losses2, f)

    with open(f'small2_adam_losses{i}.pkl', 'wb') as f:
        pickle.dump(adam_losses, f)