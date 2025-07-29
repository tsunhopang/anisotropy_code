import psutil
p = psutil.Process()
p.cpu_affinity([0])
from tqdm import tqdm
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jimgw.prior import MultivariateGaussianDistribution

data = np.load('./outdir/eos_samples_complete.npz')

# fetch the NEPs
Esym = data['E_sym']
Lsym = data['L_sym']
Ksym = data['K_sym']
Ksat = data['K_sat']

# calculate the covarience
samples = np.vstack([Esym, Lsym, Ksym, Ksat])
mean = np.mean(samples, axis=1)
cov = np.cov(samples)

# sanity checks
print(f"Esym = {mean[0]} +- {np.sqrt(cov[0,0])}")
print(f"Lsym = {mean[1]} +- {np.sqrt(cov[1,1])}")
print(f"Ksym = {mean[2]} +- {np.sqrt(cov[2,2])}")
print(f"Ksat = {mean[3]} +- {np.sqrt(cov[3,3])}")

print(f"The mean is {mean}")
print(f"The cov is {cov}")

# test jim prior
prior = MultivariateGaussianDistribution(
    mean=jnp.array(mean), cov=jnp.array(cov), parameter_names=['E_sym', 'L_sym', 'K_sym', 'K_sat']
)
key = jax.random.PRNGKey(42)

samples_test = prior.sample(key, 10000)
samples_test_unnamed = jnp.array(jax.tree.map(lambda k: samples_test[k], prior.parameter_names))
mean_test = jnp.mean(samples_test_unnamed, axis=1)
cov_test = jnp.cov(samples_test_unnamed)

# sanity checks
print(f"Esym = {mean_test[0]} +- {jnp.sqrt(cov_test[0,0])}")
print(f"Lsym = {mean_test[1]} +- {jnp.sqrt(cov_test[1,1])}")
print(f"Ksym = {mean_test[2]} +- {jnp.sqrt(cov_test[2,2])}")
print(f"Ksat = {mean_test[3]} +- {jnp.sqrt(cov_test[3,3])}")

logp = []
for i in tqdm(range(len(samples_test['E_sym']))):
    samples_per_i = {key: value[i] for key, value in samples_test.items()}
    logp.append(prior.log_prob(samples_per_i))
logp = np.array(logp)

# the mean is expected to be equal to
# mean = C - d/2, with C = -d/2 * np.log(2. * np.pi) - 0.5 * np.log(np.linalg.det(cov))
# var = d/2
C = -prior.n_dim / 2. * np.log(2. * np.pi) - 0.5 * np.log(np.linalg.det(cov))
print(f"Difference between the mean and the expected value = {np.mean(logp) - C + prior.n_dim / 2.}")
print(f"Difference between the variance and the expected value = {np.var(logp) - prior.n_dim / 2.}")
