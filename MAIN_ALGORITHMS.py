import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from jax import jacobian, jvp, vmap, value_and_grad, jit
from jax.flatten_util import ravel_pytree
from jax import random


def vec(A):
    """
    Stacks the entries of matrix A column-wise to form a vector.
    A: Input matrix of shape (m, n)
    
    Returns:
    Vector of length m * n
    """
    return A.T.reshape(-1)

def sketch(g_list, v):
    """
    Returns a K x K sketch.
    Assume that v is input as k x n_left x n_right.
    """
    full_sketch = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
        full_sketch += jnp.einsum('knm, ni, mj, fij -> kf', v, left, right, v)
    return full_sketch

def random_pos_def_sqrt(n, key, alpha=1.0, eigs=None):
    """
    Creates one half of a random positive semi-definite matrix with a given size and eigenvalues.
    n: size of matrix
    key: jax random key
    alpha: scaling factor
    eigs: list of eigenvalues (type = List)
    """
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(n, n)))
    s = alpha * jnp.array(eigs)
    return q @ jnp.diag(s)                                # so matrix is positive semi-definite

def KP_sum(g_list):
    """
    Returns full matrix from a list of Kronecker products.
    """
    res = 0.0
    for g in g_list:
        L, R = g["left"] @ g["left"].T, g["right"] @ g["right"].T
        KP = np.kron(L, R)
        res += KP
    return res


## SKETCH-AND-LEARN ALGORITHMS ##

def learn_G_one_layer(layers, true_G, iters, K=10, learning_rate=1e-4):
    """
    layers: list of tuples of (n_left, n_right) for each layer
    true_G: P x P full GGN matrix
    iters: number of iterations
    K: sketching dimension
    """
    G_est_initial = []                          
    for n_left, n_right in layers:
        G_est_initial.append([{"left": jnp.eye(n_left), "right": jnp.eye(n_right)}])
    
    n_left, n_right = layers[0]

    optimizer = optax.adam(learning_rate=learning_rate, b2=0.99)
    params = G_est_initial
    opt_state = optimizer.init(params)

    def loss_fn(G_est, V):
        reshaped_V = V.reshape(-1, n_left*n_right)
        sketch_true = reshaped_V @ true_G @ reshaped_V.T
        
        res = sketch(G_est[0], V)
        return jnp.mean((res - sketch_true)**2) 

    @jax.jit
    def update(params, opt_state, V):
        loss, grads = jax.value_and_grad(loss_fn)(params, V)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    key = jax.random.PRNGKey(15)
    
    for t in range(iters):  
        _, key = jax.random.split(key)
        V = jax.random.normal(key, shape=(K, n_left, n_right))
        
        params, opt_state, loss = update(params, opt_state, V)
        losses.append(loss)
   
        if t % 100 == 0:
            print(f"Iteration: {t}, Loss: {loss}")
    return params, losses


def learn_G_multiple_layers(layers, true_G, iters, K=10, learning_rate=1e-4):
    """
    layers: list of tuples of (n_left, n_right) for each layer
    true_G: P x P full GGN matrix
    iters: number of iterations
    K: sketching dimension
    """
    G_est_initial = []                          # initial guess of G (identity matrices)
    
    for n_left, n_right in layers:
        G_est_initial.append([{"left": jnp.eye(n_left), "right": jnp.eye(n_right)}])
    
    optimizer = optax.adam(learning_rate=learning_rate, b2=0.99)
    params = G_est_initial
    opt_state = optimizer.init(params)

    def loss_fn(V_blocks, V, G_est):
        sketch_true = V.T @ true_G @ V

        res = 0.0
        for i in range(len(V_blocks)):
            res += sketch(G_est[i], V_blocks[i])
      
        return jnp.mean((res - sketch_true)**2) 


    @jax.jit
    def update(params, opt_state, V_blocks, V):
        def compute_loss(params):
            return loss_fn(V_blocks, V, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    losses1, losses2 = [], []
    key = jax.random.PRNGKey(15)
    
    for t in range(iters):  
        V_blocks, holder = [], []
        _, key = jax.random.split(key)
        keys = jax.random.split(key, len(layers)+1) 
        
        coin_flip = jax.random.bernoulli(keys[0], 0.5, shape=(1,))
        # coin flip being false means we are sampling from layer 1, true means layer 2

        for i in range(len(layers)):
            n_left, n_right = layers[i]
            
            if i == coin_flip:
                block = jax.random.normal(keys[i+1], shape=(K, n_left, n_right))
            else:
                block = np.zeros((K, n_left, n_right))

            V_blocks.append(block) 
            holder.append(block.reshape(-1, n_left*n_right).T)

        V = np.concatenate(holder, axis=0)
        params, opt_state, loss = update(params, opt_state, V_blocks, V)
        
        losses.append(loss)
        if coin_flip:
            # layer 2 losses 
            losses2.append(loss)
        else:
            # layer 1 losses
            losses1.append(loss)
        
        if t % 100 == 0:
            print(f"Iteration: {t}, Loss: {loss} for layer {coin_flip}")
    return params, losses, losses1, losses2


def get_eigenvectors(G):
    """
    Return the eigenvectors and eigenvalues of the Kronecker-factored GGN matrix.
    G: list of dictionaries containing left and right Kronecker factors of each layer
    """
    block_sizes = []
    for blk in G:
        r = blk[0]["left"].shape[0]
        c = blk[0]["right"].shape[0]
        block_sizes.append(r * c)

    starts = np.cumsum([0] + block_sizes[:-1]) 
    P = sum(block_sizes)  

    eigenvecs, eigenvalues = [], []
    
    for offset, blk in zip(starts, G):
        A = blk[0]["left"] @ blk[0]["left"].T
        B = blk[0]["right"] @ blk[0]["right"].T
        u_A, s_A, _ = np.linalg.svd(A)
        u_B, s_B, _ = np.linalg.svd(B)

        for j in range(u_A.shape[0]):
            for k in range(u_B.shape[0]):
                eigenvalues.append(s_A[j] * s_B[k])           # eigenvalue
                
                u = vec(np.outer(u_B[:, k], u_A[:, j]))

                full_vec = np.zeros(P)
                full_vec[offset : offset + u.size] = u
                eigenvecs.append(full_vec)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvecs = [eigenvecs[i] for i in sorted_indices]
    eigenvalues = [eigenvalues[i] for i in sorted_indices]
    
    U = np.column_stack(eigenvecs)
    return U, eigenvalues



## SOFO ALGORITHMS FOR QUADRATIC LOSS SURFACES ##

def EIG_SOFO(K, true_G, approx_G, key, N, learning_rate=1, alpha=1e-4):
    """
    K: number of sketching vectors
    true_G: P x P full GGN matrix (for calculating losses)
    approx_G: list of dictionaries containing left and right Kronecker factors of each layer
    key: jax random key
    N: number of iterations
    """
    P = true_G.shape[0]  

    U, _ = get_eigenvectors(approx_G) 
    
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))

    for n in range(N):
        losses.append(0.5 * current_theta.T @ true_G @ current_theta)

        indices = (np.arange(K) + (K*n % P)) % P 
        v = U[:, indices]     

        c = v.T @ true_G @ v 
        g = v.T @ true_G @ current_theta   

        U_damp, s, Vt = np.linalg.svd(c)
        inverted_G = U_damp * (1 / (s + alpha * np.max(s))) @ Vt
        dtheta = v @ inverted_G @ g
        current_theta = current_theta - learning_rate * dtheta  

    return losses


def SOFO(K, sigma, key, N, learning_rate=1, alpha=1e-4):
    P = sigma.shape[0]  
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))
    
    for _ in range(N):
        losses.append(0.5 * current_theta.T @ sigma @ current_theta)

        v = np.random.randn(P, P)  
        v = v[:, :K]   

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ current_theta   
        
        U_damp, s, Vt = np.linalg.svd(c)
        inverted_G = U_damp * (1 / (s + alpha * np.max(s))) @ Vt
        dtheta = v @ inverted_G @ g
        current_theta = current_theta - learning_rate * dtheta 

    return losses




## NEURAL NETWORK ALGORITHMS ##
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



## SOFO ALGORITHMS FOR NEURAL NETWORKS ##
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


def nn_EIG_SOFO(K, N, initial_params, x, y, approx_G, learning_rate=1, damping=False, alpha=1e-4):
    """
    K = sketching dimension
    N = number of iterations
    initial_params = initial parameters in the form of a pytree
    x = input data
    y = labels for input data
    approx_G = list of dictionaries containing left and right Kronecker factors of each layer
    learning_rate = step size
    """
    losses = []

    U, _ = get_eigenvectors(approx_G)  
    
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




## OTHER OPTIMIZATION ALGORITHMS ##
def gradient_descent(N, initial_params, x, y, learning_rate=1):
    """
    N = number of iterations
    initial_params = initial parameters in the form of a pytree
    x = input data
    y = labels for input data
    learning_rate = step size
    """
    losses = []
    
    current_theta, unravel_fn = ravel_pytree(initial_params)    # flatten the parameters into a 1D array

    grad_loss_fn = value_and_grad(MSE_loss, argnums=0)

    for n in range(N):  
        loss, gradient = grad_loss_fn(current_theta, unravel_fn, x, y)      # gradient of cost function
        losses.append(loss)
    
        current_theta = current_theta - learning_rate * gradient

        if n % 100 == 0:
            print(f"Step {n} | Loss: {loss:.6f}")
    return losses


def adam_training(N, initial_params, x, y, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Train for N iterations with Adam.
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
