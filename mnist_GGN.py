import jax.numpy as jnp
import jax
import optax
from flax import linen as nn
from flax.training import train_state
from datasets import mnist
import numpy as np
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from jax import random
from jax import grad, value_and_grad, jvp


def random_params(di, Nh, do, key):
    """Initialise parameters of the shallow neural network."""
    w_key, c_key = random.split(key)
    C = 1/jnp.sqrt((di+1)) * random.normal(c_key, (Nh, di + 1)) 
    W = 1/jnp.sqrt(Nh) * random.normal(w_key, (do, Nh)) 
    return {"C": C, "W": W}


def nn_output(params, x):
    """Compute the output of the neural network using params tree."""
    x = jnp.vstack((x, jnp.ones((1, x.shape[1]))))           # Add 1 as the bias input
    h = jax.nn.relu(jnp.matmul(params["C"], x)) 
    y = jnp.matmul(params["W"], h)  
    return y       


def nn_output_flat(flat_params, unravel_fn, x):
    """Compute the output of the neural network using flattened params."""
    params = unravel_fn(flat_params)
    return nn_output(params, x)        # shape: (do, no_of_samples)


def cross_entropy_loss_fn(flat_params, unravel_fn, x, y_true):
    """Compute the cross entropy loss using flat params."""
    logits = nn_output_flat(flat_params, unravel_fn, x)             # get logits (10 x no_of_samples)
    log_probs = jax.nn.log_softmax(logits)                          # predicted probabilities
    loss = -jnp.mean(jnp.sum(y_true * log_probs, axis=0))
    return loss


def accuracy(flat_params, unravel_fn, x, y_true):
    """Compute the accuracy of the model."""
    logits = nn_output_flat(flat_params, unravel_fn, x)
    preds = jnp.argmax(logits, axis=0)
    labels = jnp.argmax(y_true, axis=0)
    return jnp.mean(preds == labels)


def Jv_fn(flat_params, unravel_fn, x, V):
    """Compute J @ V using jvp, one column at a time."""
    def model_fn(p):
        return nn_output_flat(p, unravel_fn, x).ravel()  # output shape (do * N,)

    Jv_columns = []
    for i in range(V.shape[1]):
        _, jv = jvp(model_fn, (flat_params,), (V[:, i],))
        Jv_columns.append(jv)
    return jnp.stack(Jv_columns, axis=1)  # shape: (do * N, K)


## Network parameters ##
di = 784                 # input dimension (28x28 images)
Nh = 100                 # number of hidden neurons
do = 10                  # output dimension (10 classes for MNIST)

## Dataset ##
train_images, train_labels, test_images, test_labels = mnist()
# train images is 60,000 x 784
# train labels is 60,000 x 10, they are one-hot encoded


## Initialise parameters ##
key = random.PRNGKey(0)
params = random_params(di, Nh, do, key)
flat_params, unravel_fn = ravel_pytree(params)


## Set up Adam
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(flat_params)

## Gradient function ##  
loss_grad_fn = value_and_grad(cross_entropy_loss_fn, argnums=0)


num_epochs = 10
batch_size = 128
num_batches = train_images.shape[0] // batch_size

train_images = train_images.T / 255.0
train_labels = train_labels.T

for epoch in range(num_epochs):
    for i in range(num_batches):
        batch_x = train_images[:, i*batch_size:(i+1)*batch_size]
        batch_y = train_labels[:, i*batch_size:(i+1)*batch_size]

        loss, grads = loss_grad_fn(flat_params, unravel_fn, batch_x, batch_y)

        updates, opt_state = optimizer.update(grads, opt_state, flat_params)
        flat_params = optax.apply_updates(flat_params, updates)

    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")



# Evaluate on test set
test_x = test_images.T / 255.0
test_y = test_labels.T
acc = accuracy(flat_params, unravel_fn, test_x, test_y)
print(f"Test accuracy: {acc:.4f}")





def H_logits_blocks1(logits):
    """Compute per-sample Hessian blocks of softmax cross-entropy."""
    probs = jax.nn.softmax(logits, axis=0)  # shape: (do, N)
    no_of_samples = probs.shape[1]
    
    H_blocks = []
    for i in range(no_of_samples):
        p = probs[:, i]
        H_i = jnp.diag(p) - jnp.outer(p, p)   # shape: (do, do)
        H_blocks.append(H_i)
    return H_blocks  # list of N (do x do) arrays


def H_logits_blocks(logits):
    """Compute (N, do, do) array of per-sample Hessians for softmax cross-entropy."""
    probs = jax.nn.softmax(logits, axis=0)  # shape: (do, N)

    def single_block(p):
        return jnp.diag(p) - jnp.outer(p, p)  # (do, do)

    H_all = jax.vmap(single_block, in_axes=1, out_axes=0)(probs)  # (N, do, do)
    return H_all



def sketched_GGN_cross_entropy1(flat_params, unravel_fn, x, V):
    logits = nn_output_flat(flat_params, unravel_fn, x)  # shape: (do, N)
    H_blocks = H_logits_blocks(logits)

    Jv = Jv_fn(flat_params, unravel_fn, x, V)  
   
    no_of_samples = logits.shape[1]
    
    G_sketched = jnp.zeros((V.shape[1], V.shape[1]))
    for i in range(no_of_samples):
        Ji = Jv[do*i : do*(i+1), :]  # shape: (do, K)
        H_i = H_blocks[i]            # shape: (do, do)
        G_sketched += Ji.T @ H_i @ Ji

    return G_sketched / no_of_samples


def sketched_GGN_cross_entropy(flat_params, unravel_fn, x, V):
    logits = nn_output_flat(flat_params, unravel_fn, x)  # (do, N)
    H_all = H_logits_blocks(logits)  # (N, do, do)

    Jv = Jv_fn(flat_params, unravel_fn, x, V)  # shape: (do * N, K)
    do, N = logits.shape
    Jv = Jv.reshape(do, N, -1).transpose(1, 0, 2)  # → shape: (N, do, K)

    # Now vectorized computation: Ji^T @ Hi @ Ji for each i
    def per_sample_block(H_i, J_i):
        return J_i.T @ H_i @ J_i  # (K, K)

    blocks = jax.vmap(per_sample_block)(H_all, Jv)  # shape: (N, K, K)

    G_sketched = jnp.mean(blocks, axis=0)  # average over N
    return G_sketched


"""
P = flat_params.shape[0]    # number of parameters
K = 100

for _ in range(3):
    V = np.random.randn(P, K)  
    G_sketched = sketched_GGN_cross_entropy(flat_params, unravel_fn, test_x, V)  # shape: (K, K)
    _, s, _ = jnp.linalg.svd(G_sketched)     
    plt.plot(s**2)

plt.title("Eigenvalues of Different Sketches of MNIST GGN")
plt.xlabel("Index")
plt.ylabel("Eigenvalues")
plt.yscale('log')
plt.show()


plt.figure(figsize=(8, 6))
plt.imshow(G_sketched, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f'Sketched GGN Matrix')
plt.xlabel('Parameters')
plt.ylabel('Parameters')
plt.show()
"""


def sketch3(g_list, v):
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

def identity_guess(n_left, n_right, scaling_factor=1.0):
    """
    Returns an initial guess of G as the identity matrix.
    The scaling factor is so the Frobenius norm of the scaled identity matrix is of order of G.
    """
    return {
        "left": jnp.eye(n_left)*scaling_factor,
        "right": jnp.eye(n_right)*scaling_factor
    }


def learn_G_big(layers, flat_params, unravel_fn, x, iters, K=10):
    """
    layers: list of tuples of (n_left, n_right) for each layer
    true_G: P x P full GGN matrix
    iters: number of iterations
    K: sketching dimension
    """
    G_est_initial = []                          # initial guess of G (identity matrices)
    
    for n_left, n_right in layers:
        G_est_initial.append([identity_guess(n_left, n_right)])
    
    optimizer = optax.adam(learning_rate=1e-4, b2=0.99)
    params = G_est_initial
    opt_state = optimizer.init(params)


    def loss_fn(V_blocks, V, G_est, sketched_GGN):

        res = 0.0
        for i in range(len(V_blocks)):
            res += sketch3(G_est[i], V_blocks[i])
      
        return jnp.mean((res - sketched_GGN)**2) 


    @jax.jit
    def update(params, opt_state, V_blocks, V, sketched_GGN):
        def compute_loss(params):
            return loss_fn(V_blocks, V, params, sketched_GGN)

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

        sketched_GGN = sketched_GGN_cross_entropy(flat_params, unravel_fn, x, V)

        params, opt_state, loss = update(params, opt_state, V_blocks, V, sketched_GGN)
        
        
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

layers = [(Nh, di+1), (do, Nh)]          
learned_G, losses, losses1, losses2 = learn_G_big(layers, flat_params, unravel_fn, test_x, iters=10000, K=512)

plt.plot(losses1)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Layer 1')
plt.show()

plt.plot(losses2)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Layer 2')
plt.show()

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Total Loss')
plt.yscale('log')
plt.show()