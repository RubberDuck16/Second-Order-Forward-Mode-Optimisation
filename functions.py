import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt


def random_pos_def_sqrt(n, key, alpha=1.0, eigs=None):
    """
    n is size of matrix
    key is a jax random key
    alpha is a scaling factor
    eigs is a list of eigenvalues (type = List)
    """
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(n, n)))
    if eigs is None:
        s = alpha * jnp.exp(-jnp.arange(n) / (n / 4))   
    else:
        s = alpha * jnp.array(eigs)
    return q @ jnp.diag(jnp.sqrt(s))                                # so matrix is positive semi-definite


def initialise_g(n_left, n_right, key, alpha=1.0, eigs_left=None, eigs_right=None):
    """
    eigs_left and eigs_right are lists of eigenvalues for the left and right matrices respectively.
    """
    key_left, key_right = jax.random.split(key)         # split key so that we get a different random matrix for left and right
    return {
        "left": random_pos_def_sqrt(n_left, key_left, alpha, eigs_left),
        "right": random_pos_def_sqrt(n_right, key_right, alpha, eigs_right)
    }


def identity_guess(n_left, n_right, scaling_factor=1.0):
    """
    Returns an initial guess of G as the identity matrix.
    The scaling factor is so the Frobenius norm of the scaled identity matrix is of order of G.
    """
    return {
        "left": jnp.eye(n_left)*scaling_factor,
        "right": jnp.eye(n_right)*scaling_factor
    }


def vec(A):
    """
    Stacks the entries of matrix A column-wise to form a vector.
    A: Input matrix of shape (m, n)
    
    Returns:
    Vector of length m * n
    """
    return A.T.reshape(-1)


def row_vec(A):
    """
    Returns the row-major vectorization of A.
    Converts a m x n matrix to a mn x 1 vector.
    """
    return A.reshape(-1, order='C')


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


def Gv_product(g_list, v):
    full_product = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
        full_product += jnp.einsum('ij,ab,kja->kib', left, right, v)
    return full_product


"""
def reshape(v, shape):
    return v.reshape(shape)

# This is the sketch function with all reshaped laid out (no einsum)
def sketch(g_list, v):   
    #v is a k x n_left x n_right tensor
    full_sketch = 0.0
    for g in g_list:
        n_left = g["left"].shape[0]
        n_right = g["right"].shape[0]

        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T
      
        gv = (reshape(v, (-1, n_right)) @ right)    # reshape v into (k*n_left, n_right) and multiply by right
        gv = reshape(gv, (-1, n_left, n_right))     # reshape gv into (k, n_left, n_right)
        
        gv = gv.transpose(0, 2, 1)                  # transpose gv to (k, n_right, n_left)
        gv = (reshape(gv, (-1, n_left)) @ left)     # reshape gv into (k*n_right, n_left) and multiply by left
        gv = reshape(gv, (-1, n_right, n_left))     # reshape gv into (k, n_right, n_left)
        gv = gv.transpose(0, 2, 1)                  # transpose gv back to (k, n_left, n_right)

        gv = reshape(gv, (-1, n_left * n_right))    # reshape gv into (k, n_left*n_right)
        v_flat = reshape(v, (-1, n_left * n_right)) # reshape v into (k, n_left*n_right) in order t multiply with gv
        
        full_sketch += v_flat @ gv.T
    return full_sketch 
"""

def KP_sum(g_list):
    """
    Returns full matrix from a list of Kronecker products.
    """
    res = 0.0
    for g in g_list:
        left = g["left"] @ g["left"].T
        right = g["right"] @ g["right"].T

        KP = np.kron(left, right)
        res += KP
    return res


def optimise_G_hat(initial_guess, true_G, K=10, iters=20000):
    """
    Initial_guess: list of dictionaries containing left and right matrices, our initial apprxo. of G as just one KP in a list
    True_G: list of dictionaries containing left and right matrices, the true G is a sum of Kronecker products in a list
    K: sketching dimension
    Iters: number of iterations
    """
    n_left = initial_guess[0]["left"].shape[0]
    n_right = initial_guess[0]["right"].shape[0]

    optimizer = optax.adam(learning_rate=1e-4, b2=0.99)
    params = initial_guess
    opt_state = optimizer.init(params)

    def loss_fn(v, r, current_g):
        if type(true_G) is list:
            sketch_true = sketch3(true_G, v)
        else:
            # ground truth GGN is not a Kronecker product
            sketch_true = v.reshape(-1, n_left*n_right) @ true_G @ v.reshape(-1, n_left*n_right).T
        sketch_approx = sketch3(current_g, v)
        frobenius_norm = jnp.linalg.norm(sketch_true, 'fro')
        return jnp.mean((sketch_approx - sketch_true) ** 2) / frobenius_norm

    @jax.jit
    def update(params, opt_state, v, r):
        def compute_loss(params):
            return loss_fn(v, r, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    losses = []
    for t in range(iters):
        key = jax.random.PRNGKey(t)
        v = jax.random.normal(key, shape=(K, n_left, n_right))
        
        # find r
        if t == 0:
            # find the scalar which we will multiply all subsequent guesses by to make them of the order of GGN
            if type(true_G) is list:
                sketch_true = sketch3(true_G, v)
            else:
                sketch_true = v.reshape(-1, n_left*n_right) @ true_G @ v.reshape(-1, n_left*n_right).T
            sketch_guess = sketch3(initial_guess, v)
            
            true_size = jnp.linalg.norm(sketch_true, 'fro')
            guess_size = jnp.linalg.norm(sketch_guess, 'fro')
            r = true_size / guess_size
            
        params, opt_state, loss = update(params, opt_state, v, r)
        losses.append(loss)
        if t % 50 == 0:
            print(f"Iteration: {t}, Loss: {loss}")
    return params, losses


def ground_truth_G(P, key=None, eigs=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(P, P)))
    if eigs is None:
        s = jnp.exp(-jnp.arange(P) / (P / 4))   
    else:
        s = jnp.array(eigs)
    return q.T @ jnp.diag(s) @ q




def learn_G(layers, true_G, iters, K=10):
    """
    layers: list of tuples of (n_left, n_right) for each layer
    true_G: P x P full GGN matrix
    iters: number of iterations
    K: sketching dimension
    """
    G_est_initial = []                          # initial guess of G (identity matrices)
    
    for n_left, n_right in layers:
        G_est_initial.append([identity_guess(n_left, n_right)])
    
    optimizer = optax.adam(learning_rate=1e-3, b2=0.99)
    params = G_est_initial
    opt_state = optimizer.init(params)

    def loss_fn(V_blocks, V, G_est):
        sketch_true = V.T @ true_G @ V 
        res = 0.0
        for i in range(len(V_blocks)):
            res += sketch3(G_est[i], V_blocks[i])
        diff = res - sketch_true
        #return jnp.linalg.norm(diff, 'fro') / jnp.linalg.norm(sketch_true, 'fro')
        return jnp.mean(diff ** 2)

    # jax select 

    @jax.jit
    def update(params, opt_state, V_blocks, V):
        def compute_loss(params):
            return loss_fn(V_blocks, V, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    # training loop
    losses = []

    key = jax.random.PRNGKey(0)
    for t in range(iters):  
        V_blocks, holder = [], []
        _, key = jax.random.split(key)
        keys = jax.random.split(key, len(layers)+1) 
        
        coin_flip = jax.random.bernoulli(keys[0], 0.5, shape=(1,))

        for i in range(len(layers)):
            n_left, n_right = layers[i]
            block = jax.random.normal(keys[i], shape=(K, n_left, n_right))
            if coin_flip == i:
                block = np.zeros((K, n_left, n_right))
            
            V_blocks.append(block) 
 
            holder.append(block.reshape(-1, n_left*n_right).T)
            
        V = np.concatenate(holder, axis=0)
        params, opt_state, loss = update(params, opt_state, V_blocks, V)
        
        losses.append(loss)
        if t % 50 == 0:
            print(f"Iteration: {t}, Loss: {loss}")
    return params, losses


def learn_G2(layers, true_G, iters, K=10):
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

    def loss_fn(V_blocks, G_est):
        res, sketch_true = 0.0, 0.0
        for i in range(len(V_blocks)):
            res += sketch3(G_est[i], V_blocks[i])
            sketch_true += sketch3(true_G[i], V_blocks[i])
        diff = res - sketch_true
        #return jnp.linalg.norm(diff, 'fro')
        return jnp.mean(diff ** 2)


    @jax.jit
    def update(params, opt_state, V_blocks):
        def compute_loss(params):
            return loss_fn(V_blocks, params)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    # training loop
    losses = []

    key = jax.random.PRNGKey(5)
    for t in range(iters):  
        V_blocks = []
        _, key = jax.random.split(key)
        keys = jax.random.split(key, len(layers)+1) 
        
        coin_flip = jax.random.bernoulli(keys[0], 0.5, shape=(1,))
        #if t % 2 == 0:
            #coin_flip = 1
        #else:
            #coin_flip = 0

        for i in range(len(layers)):
            n_left, n_right = layers[i]
            block = jax.random.normal(keys[i+1], shape=(K, n_left, n_right))
            if coin_flip == i:
                store = i
                block = np.zeros((K, n_left, n_right))
            V_blocks.append(block) 
    
        params, opt_state, loss = update(params, opt_state, V_blocks)
        
        losses.append(loss)
        if t % 50 == 0:
            print(f"Iteration: {t}, Loss: {loss} for layer {store}")
    return params, losses


def learn_G3(layers, true_G, iters, K=10):
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

    def loss_fn(V_blocks, V, G_est):
        sketch_true = V.T @ true_G @ V

        res = 0.0
        for i in range(len(V_blocks)):
            res += sketch3(G_est[i], V_blocks[i])
      
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
    Get the eigenvalues of the Kronecker product of the blocks in G
    G: list of dictionaries containing left and right matrices (learned KP approximation of G)
    """
    A = G[0][0]["left"] @ G[0][0]["left"].T
    B = G[0][0]["right"] @ G[0][0]["right"].T
    C = G[1][0]["left"] @ G[1][0]["left"].T
    D = G[1][0]["right"] @ G[1][0]["right"].T

    u_A, s_A, _ = np.linalg.svd(A)
    u_B, s_B, _ = np.linalg.svd(B)
    u_C, s_C, _ = np.linalg.svd(C)
    u_D, s_D, _ = np.linalg.svd(D)

    # Pairwise Kronecker product of each eigenvector in u_A and u_B
    kronecker_products = []
    eigenvalues = []
    for i in range(u_A.shape[0]):
        for j in range(u_B.shape[0]):
            l = s_A[i] * s_B[j]                             # eigenvalue
            u = vec(np.outer(u_B[:, j], u_A[:, i]))
            zeros = np.zeros(u_C.shape[0] * u_D.shape[0])
            u = np.concatenate((u, zeros), axis=0)              # P x 1 vectors 
            kronecker_products.append(u)
            eigenvalues.append(l)
    
    for i in range(u_C.shape[0]):
        for j in range(u_D.shape[0]):
            l = s_C[i] * s_D[j]             # eigenvalue
            u = vec(np.outer(u_D[:, j], u_C[:, i]))
            zeros = np.zeros(u_A.shape[0] * u_B.shape[0])
            u = np.concatenate((zeros, u), axis=0)              # P x 1 vectors 
            kronecker_products.append(u)
            eigenvalues.append(l)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    kronecker_products = [kronecker_products[i] for i in sorted_indices]
    eigenvalues = [eigenvalues[i] for i in sorted_indices]
    eigenvectors = np.column_stack(kronecker_products)
    return eigenvectors, eigenvalues




def learn_G_big(layers, true_G, iters, K=10):
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

    def loss_fn(V_blocks, V, G_est):
        sketch_true = V.T @ true_G @ V

        res = 0.0
        for i in range(len(V_blocks)):
            res += sketch3(G_est[i], V_blocks[i])
      
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



# FOR ONLY ONE LAYER
def learn_G_big1(layers, true_G, iters, K=10):
    """
    layers: list of tuples of (n_left, n_right) for each layer
    true_G: P x P full GGN matrix
    iters: number of iterations
    K: sketching dimension
    """
    G_est_initial = []                          # initial guess of G (identity matrices)
    
    for n_left, n_right in layers:
        G_est_initial.append([identity_guess(n_left, n_right)])
    
    n_left, n_right = layers[0]

    optimizer = optax.adam(learning_rate=1e-4, b2=0.99)
    params = G_est_initial
    opt_state = optimizer.init(params)

    def loss_fn(G_est, V):
        reshaped_V = V.reshape(-1, n_left*n_right)
        sketch_true = reshaped_V.T @ true_G @ reshaped_V
        # sketch_true = reshaped_V @ true_G @ reshaped_V.T
        
        res = sketch3(G_est[0], V)
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
