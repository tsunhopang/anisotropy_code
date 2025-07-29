import jax
import numpy as np 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import harmonic as hm

def evidence_calculation(posterior_samples, log_prob, seed=42):
    print(f"Starting Bayesian evidence calculation with harmonic")
    print(f"Permuting chains with seed {seed}...")
    np.random.seed(seed)
    permutation = np.random.permutation(posterior_samples.shape[1])
    posterior_samples = posterior_samples[:, permutation]
    log_prob = log_prob[permutation]

    ndim = posterior_samples.shape[0]
    # Spline params
    n_layers = 2
    n_bins = 64
    hidden_size = [64, 64]
    spline_range = (-10.0, 10.0)
    # Optimizer params
    learning_rate = 1e-4
    momentum = 0.9
    standardize = True
    epochs_num = 100
    # aux parameters
    nchains = 20
    training_proportion = 0.5
    temperature = 0.4

    model = hm.model.RQSplineModel(
        ndim,
        n_layers=n_layers,
        n_bins=n_bins,
        hidden_size=hidden_size,
        spline_range=spline_range,
        standardize=standardize,
        learning_rate=learning_rate,
        momentum=momentum,
        temperature=temperature,
    )

    print("Configure chains...")
    chains = hm.Chains(ndim)
    chains.add_chain(posterior_samples.T, log_prob)
    chains.split_into_blocks(nchains)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=training_proportion
    )
    print("Fit model for {} epochs...".format(epochs_num))
    model.fit(
        chains_train.samples,
        epochs=epochs_num,
        verbose=True,
        key=jax.random.PRNGKey(seed),
    )
    model.temperature = temperature
    print("Compute evidence...")
    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    ev.check_basic_diagnostic()
    err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()
    print("The Log-Bayesian-Evidence:")
    print("log_Z = {} +{} -{}".format(-ev.ln_evidence_inv, err_ln_inv_evidence[1], -err_ln_inv_evidence[0]))
    print("kurtosis = {}".format(ev.kurtosis), " Aim for ~3.")
    print("Aim for sqrt( 2/(n_eff-1) ) = {}".format(np.sqrt(2.0 / (ev.n_eff - 1))))
    check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
    print("sqrt(evidence_inv_var_var) / evidence_inv_var = {}".format(check))

    return -ev.ln_evidence_inv, err_ln_inv_evidence[1]
