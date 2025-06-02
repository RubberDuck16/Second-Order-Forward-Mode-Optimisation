import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from datasets import mnist
import numpy as np
from jax.tree_util import *
from jax import random
from Utils.functions import *

train_images, train_labels, test_images, test_labels = mnist()

def jmp(f, W, M):
    "vmapped function of jvp for Jacobian-matrix product"
    _jvp = lambda s: jax.jvp(f, (W,), (s,))     # f is function to be differentiated, (W,) are primals, (s,) are tangents
    return jax.vmap(_jvp)(M)

def ggn(tangents, h):
    Jgh = (tangents @ h)[:, None]            
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T

def random_split_like_tree(rng_key, target=None, treedef=None):
    "split key for a key for every leaf"
    if treedef is None:
        treedef = tree_structure(target)
    keys = random.split(rng_key, treedef.num_leaves)
    return tree_unflatten(treedef, keys)

def sample_v(rng, params, tangent_size):
    keys_tree = random_split_like_tree(rng, params)
    v = tree_map(
        lambda x,k: random.normal(k, (tangent_size,) + x.shape, x.dtype),         
        params, keys_tree
    )
    
    #normalize, tangent-wise
    l2 = jnp.sqrt(sum(tree_leaves(
            jax.vmap(lambda v: tree_map(lambda x: jnp.sum(jnp.square(x)),v))(v)
        )))
    v = tree_map(lambda x: jax.vmap(lambda a,b:a/b)(x,l2), v)       # divide each element of x by l2, where each element of x is a tangent vector
    return v

def blockwise_sample_v(rng, params, tangent_size, layer_idx):
    """
    Create a tree-like v with random entries for only one layer (weights + bias),
    and zeros for all other layers.

    Args:
        rng: PRNG key
        params: PyTree of model params (root has 'params' key).
        tangent_size: Number of tangents
        layer_idx: Which layer to sample for (e.g., 0 for first hidden layer)
        layer_shapes: List of (n_left, n_right) for each layer

    Returns:
        PyTree of tangents with non-zero entries only in the specified layer.
    """

    keys_tree = random_split_like_tree(rng, params['params'])

    # Helper to generate blockwise tangent for a specific parameter (kernel or bias)
    def make_blockwise_tangent(param, key, layer_name, param_name):
        if layer_name != f"Dense_{layer_idx}":
            # Not the target layer — return zero tangents
            return jnp.zeros((tangent_size,) + param.shape, dtype=param.dtype)

        if param_name == 'kernel':
            # Random tangent for weight matrix
            tangent = random.normal(key, (tangent_size,) + param.shape, dtype=param.dtype)
        elif param_name == 'bias':
            # Random tangent for bias
            tangent = random.normal(key, (tangent_size,) + param.shape, dtype=param.dtype)
        else:
            raise ValueError(f"Unknown param type: {param_name}")
        return tangent

    # Create a new PyTree where all but one block is zero
    v = {'params': {}}
    for layer_name, layer_params in params['params'].items():
        v['params'][layer_name] = {}
        for param_name, param in layer_params.items():
            v['params'][layer_name][param_name] = make_blockwise_tangent(
                param, keys_tree[layer_name][param_name], layer_name, param_name
            )

    # Normalize all tangents across the tree
    l2 = jnp.sqrt(sum(tree_leaves(jax.vmap(lambda v: tree_map(lambda x: jnp.sum(jnp.square(x)), v))(v))))
    v = tree_map(lambda x: jax.vmap(lambda a, b: a / b)(x, l2), v)
    return v



def sample_v_ggn(rng, ggn_approx, tangent_size):
    """
    Generates a single sketching matrix `V` per layer for the GGN approximation.

    Args:
        rng: JAX random number generator key.
        ggn_approx: Dictionary containing Kronecker factors for each layer.
        tangent_size: Number of tangent vectors to sample.

    Returns:
        v_blocks: Dictionary with sampled and normalized sketching matrices per layer.
    """
    v_blocks = {}

    for layer in ggn_approx.keys():
        rng, key = random.split(rng)  # Generate a new random key for this layer

        # Determine the shape of the Kronecker factor (largest dimension per layer)
        n_left = ggn_approx[layer]["A"].shape[0]  # Output size
        n_right = ggn_approx[layer]["B"].shape[0]  # Input size (+ bias)

        # Sample a random matrix V of appropriate shape
        v = random.normal(key, (tangent_size, n_left, n_right))

        # Normalize each tangent vector
        l2_norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(1, 2), keepdims=True))
        v /= l2_norm  

        # Store in dictionary
        v_blocks[layer] = v

    return v_blocks



def batch_generator(images, labels, batch_size=128):
    data_size = images.shape[0]
    indices = np.arange(data_size)     # get indices of data 
    np.random.shuffle(indices)         # shuffle indices

    # split into random batches of size batch_size
    for start_idx in range(0, data_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield images[batch_indices], labels[batch_indices]


def identity_guess(n_left, n_right, scaling_factor=1.0):
    """
    Returns an initial guess of G as the identity matrix.
    The scaling factor is so the Frobenius norm of the scaled identity matrix is of order of G.
    """
    return {
        "left": jnp.eye(n_left)*scaling_factor,
        "right": jnp.eye(n_right)*scaling_factor
    }



class MLP(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(10)(x)                 # 10 output classes for MNIST
        return x

## network goes 784 -> 100 -> 10

layers = [100,]        # one hidden layer of size 100 units
tangent_size = 512
batch_size = 128
input_size = 784
ouput_size = 10

model = MLP(hidden_sizes=layers)        # initialise MLP model

rng = random.PRNGKey(0)
rng, subkey1, subkey2 = random.split(rng, 3)

params = model.init(subkey1, random.normal(subkey2, (batch_size, input_size)))
# params is a dictionary {'params': {'Dense_0': {'kernel': (784, 100), 'bias': (100,)}, 'Dense_1': ...}}

#print(params)

num_params = np.sum(tree_leaves(tree_map(lambda x: np.prod(x.shape), params)))
print("Num Params: {}, Tangents: {} ({:.4f}%)".format(num_params, tangent_size, tangent_size/num_params*100))

v = sample_v(rng, params, tangent_size)
# v is a dictionary with same structure as params

batch_x, _ = next(batch_generator(train_images, train_labels, batch_size))      # gives batch_x of (128, 784)

output_fn = lambda params: model.apply(params, batch_x)         # get output of model for batch_x
outs, tangents_out = jmp(output_fn, params, v)  

vggv = jnp.mean(jax.vmap(ggn, in_axes=(1,0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0)
print(vggv.shape)




ggn_approx = {
    "layer_1": {
        "A": jnp.eye(100, 100),  # Output-side Kronecker factor
        "B": jnp.eye(785, 785)   # Input-side Kronecker factor
    },
    "layer_2": {
        "A": jnp.eye(10, 10),   # Output-side Kronecker factor
        "B": jnp.eye(101, 101)  # Input-side Kronecker factor
    }
}

optimizer = optax.adam(learning_rate=1e-3, b2=0.99)
opt_state = optimizer.init(ggn_approx)

def loss_fn(sampled_v, sampled_block_v, ggn_approx):
    # get a new sketch of the GGN
    output_fn = lambda params: model.apply(params, batch_x)       
    outs, tangents_out = jmp(output_fn, params, sampled_v)  
    sketch_true = jnp.mean(jax.vmap(ggn, in_axes=(1,0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0)

    # add of the sketches of each block of the approximated GGN
    res = 0.0
    for layer, v in sampled_block_v.items():
        block_sketch = jnp.einsum('knm, ni, mj, fij -> kf', v, ggn_approx[layer]["A"], ggn_approx[layer]["B"], v)
        res += block_sketch

    diff = res - sketch_true
    mse = jnp.mean(jnp.square(diff))
    #jnp.linalg.norm(diff, 'fro') / jnp.linalg.norm(sketch_true, 'fro')
    return mse

@jax.jit
def update(ggn_approx, opt_state, sampled_v, sampled_block_v):
    def compute_loss(ggn_approx):
        return loss_fn(sampled_v, sampled_block_v, ggn_approx)

    loss, grads = jax.value_and_grad(compute_loss)(ggn_approx)
    updates, opt_state = optimizer.update(grads, opt_state)
    ggn_approx = optax.apply_updates(ggn_approx, updates)
    return ggn_approx, opt_state, loss


# training loop
losses = []

for t in range(500):  
    rng = jax.random.PRNGKey(t)
    rng, subkey1 = random.split(rng, 2)
    batch_x, _ = next(batch_generator(train_images, train_labels, batch_size)) 
    
    sampled_v = sample_v(rng, params, tangent_size)
    sampled_block_v = sample_v_ggn(subkey1, ggn_approx, tangent_size)

    ggn_aprox, opt_state, loss = update(ggn_approx, opt_state, sampled_v, sampled_block_v)
    
    losses.append(loss)
    if t % 1 == 0:
        print(f"Iteration: {t}, Loss: {loss}")






