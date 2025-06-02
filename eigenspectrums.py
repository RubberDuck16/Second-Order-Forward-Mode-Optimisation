import matplotlib.pyplot as plt
import jax.numpy as jnp

n_left = 27
n_right = 13
K = 5                       # ask guillaume
P = n_left * n_right


s1 = jnp.exp(-jnp.arange(P) / (P / 4))
s2 = jnp.exp(-jnp.arange(P) / (P / 8))
s3 = jnp.array([(0.99 ** i) for i in range(P)])
s4 = jnp.array([(0.7 ** i) for i in range(P)])

plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.plot(s3, label='s3')
plt.plot(s4, label='s4')
plt.legend()
plt.show()