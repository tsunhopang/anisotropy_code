"""
Full-scale inference: we will use jim as flowMC wrapper
"""
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
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp, logit
from jimgw.prior import UniformPrior, GaussianPrior, MultivariateGaussianDistribution, CombinePrior
from jimgw.jim import Jim
import argparse

import sys
sys.path.insert(0, '/data/gravwav/thopang/projects/post_tov/scripts/')
import utils
import evidence

import equinox as eqx

from flowjax.distributions import Normal, Transformed
from flowjax.bijections import Affine, Invert, RationalQuadraticSpline
from flowjax.flows import masked_autoregressive_flow

print(f"GPU found?: {jax.devices()}")

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hierarchical Bayesian on the pTOV parameter given normalizing flow on EOS-pTOV posterior from different observatons"
    )
    parser.add_argument("--isotropic-case", 
                        default=False,
                        action='store_true',
                        help="Whether to fix both mu_gamma and sigma_gamma to zero")
    parser.add_argument("--eos-model",
                        default='metamodel',
                        type=str,
                        help="The EOS model used in the analysis, either metamodel or metemodel+peakcse"
                        )
    parser.add_argument("--sample-GW170817", 
                        default=False,
                        action='store_true',
                        help="Whether to sample the GW170817 event")
    parser.add_argument("--GW170817-NF", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/GW170817_NF.eqx')
    parser.add_argument("--sample-GW190425", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the GW190425 event")
    parser.add_argument("--GW190425-NF", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/GW190425_NF.eqx')
    parser.add_argument("--sample-GW190814", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the GW190814 event")
    parser.add_argument("--GW190814-NF", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/GW190814_NF.eqx')
    parser.add_argument("--sample-J0030", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0030 event")
    parser.add_argument("--J0030-NF", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J0030_NF.eqx')
                        #default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J0030_complex_NF.eqx')
    parser.add_argument("--sample-J0740", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0740 event")
    parser.add_argument("--J0740-NF", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J0740_NF.eqx')
    parser.add_argument("--sample-J0437", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0437 event")
    parser.add_argument("--J0437-NF",
                        type=str,
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J0437_NF.eqx')
    parser.add_argument("--sample-radio-J1614", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the radio timing mass measurement on J1614 pulsar. Do all of them at once.")
    parser.add_argument("--J1614-NF",
                        type=str,
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J1614_NF.eqx')
    parser.add_argument("--sample-radio-J0348", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the radio timing mass measurement on J0348 pulsar. Do all of them at once.")
    parser.add_argument("--J0348-NF",
                        type=str,
                        default='/data/gravwav/thopang/projects/post_tov/hierarchical_inference/MM/nf_training/J0348_NF.eqx')
    parser.add_argument("--use-zero-likelihood", 
                        default=False,
                        action='store_true', 
                        help="Whether to use a mock log-likelihood which constantly returns 0")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--sampling-seed", 
                        type=int, 
                        default=42,
                        help="Seed used for sampling")
    ### flowMC/Jim hyperparameters
    parser.add_argument("--n-loop-training", 
                        type=int, 
                        default=20,
                        help="Number of flowMC training loops.)")
    parser.add_argument("--n-loop-production", 
                        type=int, 
                        default=50,
                        help="Number of flowMC production loops.)")
    parser.add_argument("--eps-mass-matrix", 
                        type=float, 
                        default=1e-5,
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
    np.random.seed(args.sampling_seed)  # for reproducibility
    # define the prior used for this hierarchical inference
    #prior_list = [E_sat_prior, NEP_prior]
    # NEP prior
    prior_list = []
    prior_list.append(
        UniformPrior(-16.05, -15.975, parameter_names=["E_sat"])
    )
    prior_list.append(
        UniformPrior(90., 275., parameter_names=["K_sat"])
    )
    prior_list.append(
        UniformPrior(26., 43., parameter_names=["E_sym"])
    )
    prior_list.append(
        UniformPrior(15., 182., parameter_names=["L_sym"])
    )
    prior_list.append(
        UniformPrior(-330., 520., parameter_names=["K_sym"])
    )
    # peakCSE prior
    if 'peak' in args.eos_model:
        prior_list.extend([
            UniformPrior(0.1, 1.0, parameter_names=["gaussian_peak"]),
            UniformPrior(2 * 0.16, 12 * 0.16, parameter_names=["gaussian_mu"]),
            UniformPrior(0.1 * 0.16, 5 * 0.16, parameter_names=["gaussian_sigma"]),
            UniformPrior(0.1, 1.0, parameter_names=["logit_growth_rate"]),
            UniformPrior(2 * 0.16, 35 * 0.16, parameter_names=["logit_midpoint"]),
        ])

    # Hyper-prior for pTOV parameter
    # While it's common to model hyper-priors with a Gaussian over mean and variance,
    # the pTOV framework imposes bounded parameter domains.
    # To respect these bounds, we use a truncated Gaussian instead.
    if not args.isotropic_case:
        prior_list.append(
            UniformPrior(-0.5, 0.5, parameter_names=["gamma_mu"])
        )
        # the upper limit is the standard deviation of U(-0.5, 0.5)
        prior_list.append(
            UniformPrior(0., 1. / float(jnp.sqrt(12.)), parameter_names=["gamma_sigma"])
        )
    else:
        print("Running for isotropic case, for getting the Bayesian evidence")
        prior_list.append(
            UniformPrior(-1e-4, 1e-4, parameter_names=["gamma_mu"])
        )
        # the upper limit is the standard deviation of U(-0.5, 0.5)
        prior_list.append(
            UniformPrior(0., 1e-4, parameter_names=["gamma_sigma"])
        )

    # Create the output directory if it does not exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # Copy this script to the output directory, for reproducibility later on
    shutil.copy(__file__, os.path.join(args.outdir, "backup_inference.py"))

    ##################
    ### LIKELIHOOD ###
    ##################
    nf_list = []
    flow_struct = Transformed(
        masked_autoregressive_flow(
            key=jax.random.key(42),
            base_dist=Normal(jnp.zeros(len(prior_list) - 1)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
        ),
        Invert(Affine(jnp.zeros(len(prior_list) - 1), jnp.ones(len(prior_list) - 1)))
    )
    if args.sample_GW170817:
        print("Loading the NF model for GW170817")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.GW170817_NF, flow_struct)
        )
    if args.sample_GW190425:
        print("Loading the NF model for GW190425")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.GW190425_NF, flow_struct)
        )
    if args.sample_GW190814:
        print("Loading the NF model for GW190814")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.GW190814_NF, flow_struct)
        )
    if args.sample_J0030:
        print("Loading the NF model for J0030")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.J0030_NF, flow_struct)
        )
    if args.sample_J0740:
        print("Loading the NF model for J0740")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.J0740_NF, flow_struct)
        )
    if args.sample_J0437:
        print("Loading the NF model for J0437")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.J0437_NF, flow_struct)
        )
    if args.sample_radio_J1614:
        print("Loading the NF model for J1614")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.J1614_NF, flow_struct)
        )
    if args.sample_radio_J0348:
        print("Loading the NF model for J0348")
        nf_list.append(
            eqx.tree_deserialise_leaves(args.J0348_NF, flow_struct)
        )
    # Total NF list:
    #print(f"Sanity checking: nf_list = {nf_list}")
    print(f"Number of NF models used = {len(nf_list)}")
    if 'peak' in args.eos_model:
        EOS_parameters = [
            "E_sat", "K_sat", "E_sym", "L_sym", "K_sym",
            "gaussian_peak", "gaussian_mu", "gaussian_sigma",
            "logit_growth_rate", "logit_midpoint"
            ]
    else:
        EOS_parameters = ["E_sat", "K_sat", "E_sym", "L_sym", "K_sym"]

    likelihood = utils.HierarchicalLikelihood(
        nf_list=nf_list,
        EOS_parameters=EOS_parameters,
    )
    # Define the final prior
    for i in range(len(nf_list)):
        prior_list.append(
            UniformPrior(0., 1., parameter_names=[f'gamma_u_{i}'])
        )
    prior = CombinePrior(prior_list)

    # Define Jim object
    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {
        "step_size": mass_matrix * args.eps_mass_matrix
    }
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
        **kwargs
    ) 
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

    # Get the log prob, also count number of samples from it
    log_prob = np.array(sampler_state["log_prob"])
    log_prob = log_prob.flatten()
    nb_samples_production = len(log_prob)
    total_nb_samples = nb_samples_training + nb_samples_production

    # Save the final results
    print("Saving the final results")
    np.savez(os.path.join(args.outdir, "results_production.npz"), log_prob=log_prob, **samples_named_for_saving)
    print(f"Number of samples generated in training: {nb_samples_training}")
    print(f"Number of samples generated in production: {nb_samples_production}")
    print(f"Number of samples generated: {total_nb_samples}")

    # Save the runtime to a file as well
    with open(os.path.join(args.outdir, "runtime.txt"), "w") as f:
        f.write(f"{runtime}")

    idx = np.random.choice(
        np.arange(len(log_prob)),
        size=100_000,
        replace=False,
    )
    # also correct the logprob
    chosen_samples = {k: jnp.array(v[idx]) for k, v in samples_named.items()}
    log_prob = log_prob[idx]

    # now do evidence calculation
    parameters_sampled = prior.parameter_names 
    lnZ, lnZerr = evidence.evidence_calculation(
        jnp.array([chosen_samples[key] for key in parameters_sampled]),
        log_prob,
    )
    # also calculate the log_prior
    log_prior = utils.vectorized_log_prior(
        jnp.arange(len(chosen_samples[parameters_sampled[0]])),
        {key: chosen_samples[key] for key in parameters_sampled},
        prior,
    )
    # output the result
    np.savez(
        os.path.join(args.outdir, "complete_samples.npz"),
        log_prob=log_prob,
        log_prior=log_prior,
        log_likelihood=log_prob - log_prior,
        lnZ=lnZ,
        lnZerr=lnZerr,
        **chosen_samples
    )
    print("DONE analysis script")

if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
