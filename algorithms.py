import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from collections import deque
from Utils.functions import *


def GGN(eigenvalues):
    # Return a PxP curvature matrix from a list of P eigenvalues
    P = len(eigenvalues)
    u, _ = np.linalg.qr(np.random.randn(P, P)) 
    s = np.diag(np.array(eigenvalues)) 
    return u @ s @ u.T


def original_SOFO_loss(P, K, sigma, key, N, learning_rate=1):
    losses = []
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))
    losses.append(0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min))

    for i in range(N-1):
        v, _ = np.linalg.qr(np.random.randn(P, P))  
        v = v[:, :K]   

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ (current_theta-theta_min)   
        dtheta = v @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta  

        loss = 0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min)
        losses.append(loss)
    return losses


def CT_SOFO_loss(K, sigma, key, N, learning_rate=1):
    P = sigma.shape[0]
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))
    losses.append(0.5 * (current_theta).T @ sigma @ (current_theta))

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

        losses.append(0.5 * (current_theta).T @ sigma @ (current_theta))
    return losses


def truncated_CT_SOFO_loss(P, K, sigma, key, N, truncation, learning_rate=1):
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))
    losses.append(0.5 * (current_theta).T @ sigma @ (current_theta))

    previous_vs = {i: deque(maxlen=truncation) for i in range(1, K+1)}  
    for _ in range(N-1):
        V_list = []                # holds conjugate search directions for each iteration

        v_orthog, _ = np.linalg.qr(np.random.randn(P, P))   
        for a in range(1, K+1): 
            v = v_orthog[:, a-1]     

            if previous_vs[a]:
                for vj in previous_vs[a]:
                    prod = sigma @ vj
                    num = v.T @ prod
                    denom = vj.T @ prod
                    v -= (num / denom) * vj

            previous_vs[a].append(v)
            V_list.append(v)
        
        V = np.column_stack(V_list)

        c = V.T @ sigma @ V
        g = V.T @ sigma @ current_theta   
        dtheta = V @ (np.linalg.solve(c, g))              
        current_theta = current_theta - learning_rate * dtheta     

        loss = 0.5 * (current_theta).T @ sigma @ (current_theta)
        losses.append(loss)
    return losses


def KP_truncated_CT_SOFO_loss(P, K, sigma, key, N, G_approx, damping_factor=2, learning_rate=1):
    G_l = G_approx[0]["left"] @ G_approx[0]["left"].T
    G_r = G_approx[0]["right"] @ G_approx[0]["right"].T
    n_left = G_l.shape[0]
    n_right = G_r.shape[0]
    
    losses = []
    theta_min = np.zeros(P)
    current_theta = jax.random.normal(key, shape=(P,))

    losses.append(0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min))
    print('Initial loss:', losses[0])

    truncation = min(n_left, n_right)

    previous_vs = {i: deque(maxlen=truncation) for i in range(1, K+1)}  
    for n in range(N-1):
        V_list = []                # holds conjugate search directions for each iteration

        for k in range(1, K+1):
            v_l = np.random.randn(n_left, 1)
            v_r = np.random.randn(n_right, 1)
            v_l /= np.linalg.norm(v_l)
            v_r /= np.linalg.norm(v_r)

            if previous_vs[k]:
                for vj in previous_vs[k]:
                    vj_l = vj["left"]
                    vj_r = vj["right"]
                    
                    prod = G_l @ vj_l
                    num = v_l.T @ prod
                    denom = vj_l.T @ prod
                    v_l -= (num / denom) * vj_l

                    prod = G_r @ vj_r
                    num = v_r.T @ prod
                    denom = vj_r.T @ prod
                    v_r -= (num / denom) * vj_r
                
            v = {"left": v_l, "right": v_r}
            
            previous_vs[k].append(v)
            V_list.append(np.kron(v_l, v_r)) 

        
        V = np.column_stack(V_list)
        
        c = V.T @ sigma @ V
        
        # compute SVD of sketch
        #_, s, _ = jnp.linalg.svd(c)
        #s_max = jnp.max(jnp.diag(s))

        #new_c = c + (s_max*damping_factor)*jnp.eye(c.shape[0])

        g = V.T @ sigma @ (current_theta-theta_min)            
        dtheta = V @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta     

        losses.append(0.5 * (current_theta-theta_min).T @ sigma @ (current_theta-theta_min))

    return losses


def KP_truncated_CT_SOFO_loss_changed(P, K, sigma, key, N, G_approx, learning_rate=1):
    G_l = G_approx[0]["left"] @ G_approx[0]["left"].T
    G_r = G_approx[0]["right"] @ G_approx[0]["right"].T
    n_left, n_right = G_l.shape[0], G_r.shape[0]

    sigma_left = sigma[0]["left"] @ sigma[0]["left"].T
    sigma_right = sigma[0]["right"] @ sigma[0]["right"].T
    
    losses = []    
    
    current_theta = jax.random.normal(key, shape=(P,))
    W = current_theta.reshape(n_left, n_right)              # reshape into weight matrix

    losses.append(0.5 * np.sum(W * (sigma_left @ W @ sigma_right.T)))

    truncation = min(n_left, n_right)
    previous_vs = {i: deque(maxlen=truncation) for i in range(1, K+1)}  
    for n in range(N-1):
        V_l_list = []           # should be n_left x K
        V_r_list = []           # should be n_right x K

        for k in range(1, K+1):
            v_l = np.random.randn(n_left, 1)
            v_r = np.random.randn(n_right, 1)
            v_l /= np.linalg.norm(v_l)
            v_r /= np.linalg.norm(v_r)

            if previous_vs[k]:
                for vj in previous_vs[k]:
                    vj_l = vj["left"]
                    vj_r = vj["right"]
                    
                    prod = G_l @ vj_l
                    num = v_l.T @ prod
                    denom = vj_l.T @ prod
                    v_l -= (num / denom) * vj_l
                    
                    prod = G_r @ vj_r
                    num = v_r.T @ prod
                    denom = vj_r.T @ prod
                    v_r -= (num / denom) * vj_r
                        
            previous_vs[k].append({"left": v_l, "right": v_r})
            V_l_list.append(v_l)
            V_r_list.append(v_r)

        V_l = np.column_stack(V_l_list)
        V_r = np.column_stack(V_r_list)
        
        # sketch abd loss gradient
        c = (V_l.T @ G_l @ V_l) * (V_r.T @ G_r @ V_r)      
        #g = np.array([V_l[:, i].T @ G_l @ W @ G_r.T @ V_r[:, i] for i in range(K)])     
        g = np.einsum('ni,np,pq,mq,mi->i', V_l, G_l, W, G_r, V_r)

        # (G)^(-1) * g
        #c_damped = c + 10e-16*np.eye(c.shape[0])
        b = np.linalg.solve(c, g)

        # parameter update
        #dtheta = sum(b[j] * np.outer(V_l[:, j], V_r[:, j]) for j in range(K)) # einsum
        dtheta = np.einsum('j,ij,kj->ik', b, V_l, V_r)
        W = W - learning_rate * dtheta
        
        losses.append(0.5 * np.sum(W.T * (sigma_left @ W @ sigma_right.T)))
    
    return losses



def KP_conj_tangents(layers, K, sigma, key, N, G_approx, learning_rate=1):
    """
    layers should be a list of tuples [(n_left, n_right), ...]
    sigma should be a list of lists of KP dictionaries
    same with G_approx
    """
    losses = []

    P = 0
    for n_left, n_right in layers:
        P += n_left * n_right

    current_theta = jax.random.normal(key, shape=(P,))

    Ws = []
    start_idx = 0
    for n_left, n_right in layers:
        p = n_left*n_right
        Ws.append(current_theta[start_idx:start_idx+p].reshape(n_left, n_right))
        start_idx += p

    loss = 0
    for i in range(len(layers)):
        W = Ws[i]
        sigma_left = sigma[i][0]["left"] @ sigma[i][0]["left"].T
        sigma_right = sigma[i][0]["right"] @ sigma[i][0]["right"].T
        loss += 0.5 * np.sum(W * (sigma_left @ W @ sigma_right.T))
    losses.append(loss)

    previous_vs = [{i: deque(maxlen=min(n_left, n_right)) for i in range(1, K+1)} for n_left, n_right in layers]
    for _ in range(N-1):
        V_l_lists = [[] for i in range(len(layers))]           
        V_r_lists = [[] for i in range(len(layers))]           

        for k in range(1, K+1):
            for i in range(len(layers)):
                n_left, n_right = layers[i]
                G_l = G_approx[i][0]["left"] @ G_approx[i][0]["left"].T
                G_r = G_approx[i][0]["right"] @ G_approx[i][0]["right"].T
                
                v_l = np.random.randn(n_left, 1)
                v_r = np.random.randn(n_right, 1)
                v_l /= np.linalg.norm(v_l)
                v_r /= np.linalg.norm(v_r)
                
                if previous_vs[i][k]:
                    for vj in previous_vs[i][k]:
                        vj_l = vj["left"]
                        vj_r = vj["right"]
                        
                        prod = G_l @ vj_l
                        num = v_l.T @ prod
                        denom = vj_l.T @ prod
                        v_l -= (num / denom) * vj_l
                        
                        prod = G_r @ vj_r
                        num = v_r.T @ prod
                        denom = vj_r.T @ prod
                        v_r -= (num / denom) * vj_r
                
                previous_vs[i][k].append({"left": v_l, "right": v_r})
                V_l_lists[i].append(v_l)
                V_r_lists[i].append(v_r)

        V_ls = [np.column_stack(V_l_list) for V_l_list in V_l_lists]
        V_rs = [np.column_stack(V_r_list) for V_r_list in V_r_lists]

        loss = 0
        for i in range(len(layers)):
            sigma_left = sigma[i][0]["left"] @ sigma[i][0]["left"].T
            sigma_right = sigma[i][0]["right"] @ sigma[i][0]["right"].T
            c = (V_ls[i].T @ sigma_left @ V_ls[i]) * (V_rs[i].T @ sigma_right @ V_rs[i])
            g = np.einsum('ni,np,pq,mq,mi->i', V_ls[i], sigma_left, Ws[i], sigma_right, V_rs[i])
            b = np.linalg.solve(c, g)
            dtheta = np.einsum('j,ij,kj->ik', b, V_ls[i], V_rs[i])
            Ws[i] -= 1 * dtheta 
            loss += 0.5 * np.sum(Ws[i] * (sigma_left @ Ws[i] @ sigma_right.T))
        losses.append(loss)
    
    return losses


def new_SOFO(K, sigma, U, key, N, learning_rate=1, cycle=None):
    """
    K = sketching dimension
    sigma = GGN
    U = approximated eigenvectors of GGN
    key = for initial guess
    N = number of iterations
    learning_rate = step size
    cycle = 'chunks' or 'straight' (go through eigenevectors in order)
    """
    P = sigma.shape[0]
    losses = []
    current_theta = jax.random.normal(key, shape=(P,))

    # divide P into K sections for chunking
    index = np.linspace(0, P, K+1, dtype=int)

    for n in range(N):  
        losses.append(0.5 * current_theta.T @ sigma @ current_theta)
        
        if cycle is None or cycle == 'chunks':
            indices = index + (n % int(np.ceil(P/K)))
            indices = indices[:-1]
        if cycle == 'straight':
            indices = (np.arange(K) + (K*n % P)) % P
        
        v = U[:, indices]     

        c = v.T @ sigma @ v 
        g = v.T @ sigma @ current_theta   
        dtheta = v @ (np.linalg.solve(c, g))
        current_theta = current_theta - learning_rate * dtheta  

    return losses