"""
Full-scale inference: we will use jim as flowMC wrapper
"""

# for making it works on CITs
import psutil
p = psutil.Process()
p.cpu_affinity([0])
# supress furture warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################
### PREAMBLE ###
################
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import shutil
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.prior import UniformPrior, GaussianPrior, CombinePrior
from jimgw.jim import Jim
import argparse

import utils

print(f"GPU found?: {jax.devices()}")

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--sampling-seed", 
                        type=int, 
                        default=11,
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    parser.add_argument("--use-zero-likelihood", 
                        default=False,
                        action='store_true', 
                        help="Whether to use a mock log-likelihood which constantly returns 0")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--N-samples-EOS", 
                        type=int, 
                        default=10_000, 
                        help="Number of samples for which the TOV equations are solved")
    ### flowMC/Jim hyperparameters
    parser.add_argument("--n-loop-training", 
                        type=int, 
                        default=20,
                        help="Number of flowMC training loops.)")
    parser.add_argument("--n-loop-production", 
                        type=int, 
                        default=20,
                        help="Number of flowMC production loops.)")
    parser.add_argument("--eps-mass-matrix", 
                        type=float, 
                        default=1e-3,
                        help="Overall scaling factor for the step size matrix for MALA.")
    parser.add_argument("--n-local-steps", 
                        type=int, 
                        default=2,
                        help="Number of local steps to perform.")
    parser.add_argument("--n-global-steps", 
                        type=int, 
                        default=100,
                        help="Number of global steps to perform.")
    parser.add_argument("--n-epochs", 
                        type=int, 
                        default=20,
                        help="Number of epochs for NF training.")
    parser.add_argument("--n-chains", 
                        type=int, 
                        default=1000,
                        help="Number of MCMC chains to evolve.")
    parser.add_argument("--train-thinning", 
                        type=int, 
                        default=1,
                        help="Thinning factor before feeding samples to NF for training.")
    parser.add_argument("--output-thinning", 
                        type=int, 
                        default=5,
                        help="Thinning factor before saving samples.")
    return parser.parse_args()

def main(args):

    np.random.seed(args.sampling_seed) # for reproducibility
    
    NMAX_NSAT = 3

    ### NEP priors
    # using prior from https://arxiv.org/abs/2310.11588
    prior_list = []
    E_sat_prior = GaussianPrior(-16.0, 0.005, parameter_names=["E_sat"]) # extremely tight
    K_sat_prior = GaussianPrior(230.0, 30.0, parameter_names=["K_sat"])
    #Q_sat_prior = UniformPrior(-1100.0, 2100.0, parameter_names=["Q_sat"])
    #Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    E_sym_prior = UniformPrior(24.7, 40.3, parameter_names=["E_sym"])
    L_sym_prior = UniformPrior(-11.4, 149.4, parameter_names=["L_sym"])
    K_sym_prior = UniformPrior(-400.0, 400.0, parameter_names=["K_sym"])
    #Q_sym_prior = UniformPrior(-489.0, 1223.0, parameter_names=["Q_sym"])
    #Z_sym_prior = UniformPrior(-10110.0, 2130.0, parameter_names=["Z_sym"])

    prior_list.extend([
        E_sym_prior,
        L_sym_prior, 
        K_sym_prior,
    #    Q_sym_prior,
    #    Z_sym_prior,

        E_sat_prior,
        K_sat_prior,
    #    Q_sat_prior,
    #    Z_sat_prior,
    ])

    ### CSE priors
    #if NB_CSE > 0:
    #    print(f"Using {NB_CSE} speed-of-sound extension segments")
    #    nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
    #    prior_list.append(nbreak_prior)
    #    for i in range(NB_CSE):
    #        # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
    #        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
    #        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

    #    # Final point to end
    #    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

    # Construct the EOS prior and a transform here which can be used down below for creating the EOS plots after inference is completed
    eos_prior = CombinePrior(prior_list)
    eos_param_names = eos_prior.parameter_names
    all_output_keys = [
        "n_EOS", "Esym_EOS", "Psym_EOS", "Psnm_EOS"
    ]
    name_mapping = (eos_param_names, all_output_keys)
    
    # Create the output directory if it does not exist
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Copy this script to the output directory, for reproducibility later on
    shutil.copy(__file__, os.path.join(outdir, "backup_inference.py"))

    # Now for the list of experiments considered
    # all these are gonna be hard-coded
    # sorry, future Peter
    keep_names = []

    # adding more priors
    prior_list.append(
        GaussianPrior(0.101, 0.005, parameter_names=["den_Mass_Skyrme"])   
    )
    keep_names.append("den_Mass_Skyrme")

    prior_list.append(
        GaussianPrior(0.115, 0.002, parameter_names=["den_Mass_DFT"])   
    )
    keep_names.append("den_Mass_DFT")

    prior_list.append(
        GaussianPrior(0.106, 0.006, parameter_names=["den_IAS"])   
    )
    keep_names.append("den_IAS")

    prior_list.append(
        GaussianPrior(0.035, 0.011, parameter_names=["den_HIC_Isodiff"])   
    )
    keep_names.append("den_HIC_Isodiff")

    prior_list.append(
        GaussianPrior(0.069, 0.008, parameter_names=["den_HIC_npratio"])   
    )
    keep_names.append("den_HIC_npratio")

    prior_list.append(
        GaussianPrior(0.232, 0.032, parameter_names=["den_HIC_pi"])   
    )
    keep_names.append("den_HIC_pi")
    
    ##################
    ### LIKELIHOOD ###
    ##################

    # Likelihood: choose which PSR(s) to perform inference on:
    # TODO: if using binary Love, need an extra toggle for that
    if not args.use_zero_likelihood:
        likelihoods_list_Psnm = []
        likelihoods_list_Psnm.append(
            utils.PsnmLikelihood_without_densities(
                mu=10.1, sigma=3.0,
                exp_name='HIC_DLL',
                exp_den=0.32,
            )
        )
        likelihoods_list_Psnm.append(
            utils.PsnmLikelihood_without_densities(
                mu=10.3, sigma=2.8,
                exp_name='HIC_FOPI',
                exp_den=0.32,
            )
        )

        likelihoods_list_Esym = []
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_without_densities(
                mu=15.9, sigma=1.0,
                exp_name='alphaD',
                exp_den=0.05,
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=24.7, sigma=0.8,
                exp_name='Mass_Skyrme',
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=25.4, sigma=1.1,
                exp_name='Mass_DFT',
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=25.5, sigma=1.1,
                exp_name='IAS',
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=10.3, sigma=1.0,
                exp_name='HIC_Isodiff',
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=16.8, sigma=1.2,
                exp_name='HIC_npratio',
            )
        )
        likelihoods_list_Esym.append(
            utils.EsymLikelihood_with_densities(
                mu=52.0, sigma=13.0,
                exp_name='HIC_pi',
            )
        )

        likelihoods_list_Psym = []
        #likelihoods_list_Psym.append(
        #    utils.PsymLikelihood_without_densities(
        #        mu=2.38, sigma=0.75,
        #        exp_name='PREX_II',
        #        exp_den=0.11,
        #    )
        #)
        likelihoods_list_Psym.append(
            utils.PsymLikelihood_with_densities(
                mu=10.9, sigma=8.7,
                exp_name='HIC_pi',
            )
        )
        likelihoods_list_Psym.append(
            utils.PsymLikelihood_without_densities(
                mu=12.1, sigma=8.4,
                exp_name='HIC_npflow',
                exp_den=0.24,
            )
        )

        # Total likelihoods list:
        likelihoods_list = \
            likelihoods_list_Psnm \
            + likelihoods_list_Esym \
            + likelihoods_list_Psym 
        print(f"Sanity checking: likelihoods_list = {likelihoods_list}")
        print(f"Sanity checking: prior_list = {prior_list}")
        likelihood = utils.CombinedLikelihood(likelihoods_list)
        
    # Construct the transform object
    all_output_keys = [
        "n_EOS", "Esym_EOS", "Psym_EOS", "Psnm_EOS"
    ]
    prior = CombinePrior(prior_list)
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, all_output_keys)
    my_transform = utils.MicroToBulkTransform(
        name_mapping,
        keep_names = None,
        nmax_nsat = NMAX_NSAT,
    )
    
    if args.use_zero_likelihood:
        print("Using the zero likelihood:")
        likelihood = utils.ZeroLikelihood(my_transform)

    # Define Jim object
    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * args.eps_mass_matrix}
    kwargs = {
        "n_loop_training": args.n_loop_training,
        "n_loop_production": args.n_loop_production,
        "n_chains": args.n_chains,
        "n_local_steps": args.n_local_steps,
        "n_global_steps": args.n_global_steps,
        "n_epochs": args.n_epochs,
        "train_thinning": args.train_thinning,
        "output_thinning": args.output_thinning,
    }
    
    print("We are going to give these kwargs to Jim:")
    print(kwargs)
    print("We are going to sample the following parameters:")
    print(prior.parameter_names)

    jim = Jim(
        likelihood,
        prior,
        local_sampler_arg = local_sampler_arg,
        likelihood_transforms = [my_transform],
        **kwargs
    )

    # Test case
    #samples = prior.sample(jax.random.PRNGKey(0), 3)
    #samples_transformed = jax.vmap(my_transform.forward)(samples)
    #log_prob = jax.vmap(likelihood.evaluate)(samples_transformed, {})
    #
    #print("log_prob")
    #print(log_prob)
    
    # Do the sampling
    print(f"Sampling seed is set to: {args.sampling_seed}")
    start = time.time()
    jim.sample(jax.random.PRNGKey(args.sampling_seed))
    jim.print_summary()
    end = time.time()
    runtime = end - start

    print(f"Inference has been successful, now we will do some postprocessing. Sampling time: roughly {int(runtime / 60)} mins")

    ### POSTPROCESSING ###
    # Training (just to count number of samples)
    sampler_state = jim.sampler.get_sampler_state(training=True)
    log_prob = sampler_state["log_prob"].flatten()
    nb_samples_training = len(log_prob)

    # Production (also for postprocessing plotting)
    sampler_state = jim.sampler.get_sampler_state(training=False)

    # Get the samples, and also get them as a dictionary
    samples_named = jim.get_samples()
    samples_named_for_saving = {k: np.array(v) for k, v in samples_named.items()}
    samples_named = {k: np.array(v).flatten() for k, v in samples_named.items()}
    keys, samples = list(samples_named.keys()), np.array(list(samples_named.values()))

    # Get the log prob, also count number of samples from it
    log_prob = np.array(sampler_state["log_prob"])
    log_prob = log_prob.flatten()
    nb_samples_production = len(log_prob)
    total_nb_samples = nb_samples_training + nb_samples_production
    
    # Save the final results
    print(f"Saving the final results")
    np.savez(os.path.join(outdir, "results_production.npz"), log_prob=log_prob, **samples_named_for_saving)

    print(f"Number of samples generated in training: {nb_samples_training}")
    print(f"Number of samples generated in production: {nb_samples_production}")
    print(f"Number of samples generated: {total_nb_samples}")
    
    # Save the runtime to a file as well
    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}")

    # Generate the final EOS + TOV samples from the EOS parameter samples
    idx = np.random.choice(
        np.arange(len(log_prob)),
        size=args.N_samples_EOS,
        replace=False
    )
    TOV_start = time.time()
    chosen_samples = {k: jnp.array(v[idx]) for k, v in samples_named.items()}
    # NOTE: jax lax map helps us deal with batching, but a batch size multiple of 10 gives errors, therefore this weird number
    transformed_samples = jax.lax.map(
        jax.jit(my_transform.forward),
        chosen_samples, batch_size = 4_999
    )
    TOV_end = time.time()
    print(f"Time taken for TOV map: {TOV_end - TOV_start} s")
    chosen_samples.update(transformed_samples)

    log_prob = log_prob[idx]
    np.savez(
        os.path.join(args.outdir, "eos_samples.npz"),
        log_prob=log_prob,
        **chosen_samples
    )

    for likelihood in likelihoods_list:
        first_samples = {key: value[0] for key, value in chosen_samples.items()}
        logp = likelihood.evaluate(first_samples, {})
        print(f"For {likelihood}, the logp is {logp}")

    print("DONE analysis script")
    
if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
