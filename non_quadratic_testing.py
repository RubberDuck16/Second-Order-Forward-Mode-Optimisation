import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import hessian
import jax.numpy as jnp
from Utils.functions import *
from Utils.algorithms import *


def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def loss_func(theta, code):
    if code == 'Rosenbrock':
        return rosenbrock(theta)


def CT_SOFO_loss2(K, sigma, key, N, learning_rate=1, code=None):
    P = sigma.shape[0]
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))

    if code == 'Quadratic':
        losses.append(0.5 * current_theta.T @ sigma @ current_theta)
    else:
        losses.append(loss_func(current_theta, code))

    previous_vs = {}   
    for _ in range(N-1):
        V_list = []                     # holds conjugate search directions for each iteration

        v_orthog, _ = np.linalg.qr(np.random.randn(P, P))  
        for i in range(1, K+1):
            v = v_orthog[:, i-1]

            if i in previous_vs.keys(): 
                for j in range(len(previous_vs[i])):        # check all previous search directions in this K dimension
                    vj = previous_vs[i][j]
                    num = v.T @ sigma @ vj
                    denom = vj.T @ sigma @ vj
                    v -= (num/denom) * vj
                previous_vs[i].append(v)
            else:
                previous_vs[i] = [v]            # for K dimension i, add the first search direction
            
            V_list.append(v)                # add conjugated v to the list that will hold the K v's
        V = np.column_stack(V_list)         # stack the K v's to form the V matrix

        c = V.T @ sigma @ V
        g = V.T @ sigma @ current_theta   
        dtheta = V @ (np.linalg.solve(c, g))          
        current_theta = current_theta - learning_rate * dtheta   

        if code == 'Quadratic':
            losses.append(0.5 * current_theta.T @ sigma @ current_theta)
        else:
            losses.append(loss_func(current_theta, code))
    return losses

def new_SOFO(P, K, sigma, U, key, N, learning_rate=1, code=None):
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))

    # divide P into K sections
    index = np.linspace(0, P, K+1, dtype=int)

    for n in range(N):
        if code == 'Quadratic':
            losses.append(0.5 * current_theta.T @ sigma @ current_theta)
        else:
            losses.append(loss_func(current_theta, code))
        
        indices = index + (n % int(np.ceil(P/K)))
        indices = indices[:-1]
        v = U[:, indices]     

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ current_theta   
        dtheta = v @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta  

    return losses

# NOTE: took the QR decomposition out of making the V matrix to speed things up
def SOFO(P, K, sigma, key, N, learning_rate=1, code=None):
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))
    
    for _ in range(N):
        if code == 'Quadratic':
            losses.append(0.5 * current_theta.T @ sigma @ current_theta)
        else:
            losses.append(loss_func(current_theta, code))

        v = np.random.randn(P, P)  
        v = v[:, :K]   

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ current_theta   
        dtheta = v @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta  

    return losses


def new_SOFO2(P, K, sigma, U, key, N, learning_rate=1, code=None):
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))

    for n in range(N):
        if code == 'Quadratic':
            losses.append(0.5 * current_theta.T @ sigma @ current_theta)
        else:
            losses.append(loss_func(current_theta, code))
        
        indices = (np.arange(K) + (K*n % P)) % P 
        v = U[:, indices]     

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ current_theta   
        dtheta = v @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta  

    return losses




def sigma(P, code):
    if code == 'Rosenbrock':
        # check if P is even
        if P % 2 != 0:
            raise ValueError("P must be even for Rosenbrock function.")
        theta0 = jnp.ones(P)
        return hessian(rosenbrock)(theta0)  
    
    elif code == 'Quadratic':
        # creates a quadratic loss function with a random positive definite Hessian
        alpha = 0.2
        eigs = [1/(1+i/(alpha*P)) for i in range(P)]
        M = random_pos_def_sqrt(P, jax.random.PRNGKey(1), eigs=eigs)
        return M @ M.T

P = 400
K = 4

def q(x, A):
    return 0.5 * x.T @ (A @ x)

def plot_surface(ax, func, domain, title):
    x = np.linspace(-domain, domain, 50)
    y = np.linspace(-domain, domain, 50)
    X, Y = np.meshgrid(x, y)
    stacked = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.array([func(v) for v in stacked])
    Z = Z.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return surf

# First figure: Surfaces
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax2 = fig1.add_subplot(1, 2, 2, projection='3d')

# Top left: Quadratic surface
code = 'Quadratic'
sigma_matrix = sigma(2, code=code)
plot_surface(ax1, lambda v: q(v, sigma_matrix), 2, 'Quadratic Surface')

# Top right: Rosenbrock surface
plot_surface(ax2, rosenbrock, 2, 'Rosenbrock Surface')

plt.tight_layout()
plt.show()

# Second figure: Loss plots
fig2 = plt.figure(figsize=(12, 5))
ax3 = fig2.add_subplot(1, 2, 1)
ax4 = fig2.add_subplot(1, 2, 2)

# Bottom left: Losses for quadratic
P = 400
K = 4
code = 'Quadratic'
sigma_matrix = sigma(P, code=code)
U, _, _ = np.linalg.svd(sigma_matrix)
key = jax.random.PRNGKey(14)
#losses = new_SOFO(P, K, sigma_matrix, U, key, N=500, code=code)
losses2 = SOFO(P, K, sigma_matrix, key, N=500, code=code)
losses3 = new_SOFO2(P, K, sigma_matrix, U, key, N=500, code=code)
losses4 = CT_SOFO_loss(K, sigma_matrix, key, N=500)
ax3.plot(losses2, label=f'SOFO')
#ax3.plot(losses, label=f'EIG-SOFO')
ax3.plot(losses3, label=f'EIG-SOFO')
ax3.plot(losses4, label=f'CT-SOFO')
ax3.axvline(x=P/K, color='black', linestyle='--', alpha=0.5, label='P/K')
ax3.set_xlabel('Iteration', fontsize=14)
ax3.set_ylabel('Loss', fontsize=14)
ax3.legend(fontsize=12)

# Bottom right: Losses for Rosenbrock
code = 'Rosenbrock'
P = 400
K = 4
sigma_matrix = sigma(P, code=code)
U, _, _ = np.linalg.svd(np.array(sigma_matrix))

num_trials = 5
all_losses2 = []
all_losses3 = []
all_losses4 = []

for i in range(num_trials):
    print('TRIAL ', i+1)
    key = jax.random.PRNGKey(14 + i)
    all_losses3.append(new_SOFO2(P, K, np.array(sigma_matrix), U, key, N=500, code=code))
 
avg_losses3 = np.mean(all_losses3, axis=0)

ax4.plot(SOFO(P, K, np.array(sigma_matrix), key, N=500, code=code), label='SOFO')
ax4.plot(avg_losses3, label='EIG-SOFO')
ax4.plot(CT_SOFO_loss2(K, np.array(sigma_matrix), key, N=500, learning_rate=1, code=code), label='CT-SOFO')
ax4.set_xlabel('Iteration', fontsize=14)
ax4.set_ylabel('Loss', fontsize=14)
ax4.axvline(x=P/K, color='black', linestyle='--', alpha=0.5, label='P/K')
ax4.legend(fontsize=12)

plt.tight_layout()
plt.show()
