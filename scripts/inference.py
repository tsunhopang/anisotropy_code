"""
Full-scale inference: we will use jim as flowMC wrapper
"""

# for making it works on CITs
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
from jimgw.prior import UniformPrior, GaussianPrior, MultivariateGaussianDistribution, CombinePrior
from jimgw.jim import Jim
import argparse

import sys
sys.path.insert(0, '/data/gravwav/thopang/projects/post_tov/scripts/')
import utils
import evidence

print(f"GPU found?: {jax.devices()}")

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--GR-or-nonGR", 
                        type=str, 
                        default="GR", 
                        help="Whether to run with non-GR modification. Choose from 'GR' or 'nonGR'.")
    parser.add_argument("--sample-GW170817", 
                        default=False,
                        action='store_true',
                        help="Whether to sample the GW170817 event")
    parser.add_argument("--GW170817-data", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/GW170817_NF.eqx',)
    parser.add_argument("--sample-GW190425", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the GW190425 event")
    parser.add_argument("--GW190425-data", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/GW190425_NF.eqx',)
    parser.add_argument("--sample-GW190814", 
                        default=False,
                        action='store_true',
                        help="Whether to sample the GW190814 event")
    parser.add_argument("--sample-J0030", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0030 event")
    parser.add_argument("--J0030-data-Amsterdam", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0030_Amsterdam_normal_NF.eqx', 
                        help="Path to Amsterdam's J0030 data")
    parser.add_argument("--J0030-data-Maryland", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0030_Maryland_NF.eqx', 
                        help="Path to Maryland's J0030 data")
    parser.add_argument("--sample-J0740", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0740 event")
    parser.add_argument("--J0740-data-Amsterdam", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0740_Amsterdam_NF.eqx', 
                        help="Path to Amsterdam's J0740 data")
    parser.add_argument("--J0740-data-Maryland", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0740_Maryland_NF.eqx', 
                        help="Path to Maryland's J0740 data")
    parser.add_argument("--sample-J0437", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the J0437 event")
    parser.add_argument("--J0437-data-Amsterdam",
                        type=str,
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0437_Amsterdam_NF.eqx', 
                        help="Path to Amsterdam's J0437 data")
    parser.add_argument("--J0437-data-Maryland",
                        type=str,
                        default='/data/gravwav/thopang/projects/post_tov/data/nf_training/J0437_Amsterdam_NF.eqx', 
                        help="Path to Maryland's J0437 data")
    parser.add_argument("--sample-radio-J1614", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the radio timing mass measurement on J1614 pulsar. Do all of them at once.")
    parser.add_argument("--sample-radio-J0348", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample the radio timing mass measurement on J0348 pulsar. Do all of them at once.")
    parser.add_argument("--sample-PREX", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample PREX data")
    parser.add_argument("--PREX-data", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nuclear/PREX_Esym_Lsym.dat', 
                        help="Path to PREX data")
    parser.add_argument("--sample-CREX", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample CREX data")
    parser.add_argument("--CREX-data", 
                        type=str, 
                        default='/data/gravwav/thopang/projects/post_tov/data/nuclear/CREX_Esym_Lsym.dat', 
                        help="Path to CREX data")
    parser.add_argument("--sample-chiEFT", 
                        default=False,
                        action='store_true', 
                        help="Whether to sample chiEFT data")
    parser.add_argument("--use-zero-likelihood", 
                        default=False,
                        action='store_true', 
                        help="Whether to use a mock log-likelihood which constantly returns 0")
    parser.add_argument("--fix-alpha", 
                        default=None,
                        type=float,
                        help="Value of alpha to be fixed (default: None})")
    parser.add_argument("--fix-beta", 
                        default=None,
                        type=float,
                        help="Value of beta to be fixed (default: None})")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--N-samples-EOS", 
                        type=int, 
                        default=100_000, 
                        help="Number of samples for which the TOV equations are solved")
    parser.add_argument("--nb-cse", 
                        type=int, 
                        default=8, 
                        help="Number of CSE grid points (excluding the last one at the end, since its density value is fixed, we do add the cs2 prior separately.)")
    parser.add_argument("--fix-nbreak", 
                        type=float,
                        default=None, 
                        help="Value of nbreak to fix in case CSE is in use.")
    parser.add_argument("--fix-ngrid", 
                        default=False,
                        action='store_true',
                        help="Whether to fix the density grid instead of freely moving them")
    parser.add_argument("--use-peakCSE", 
                        default=False,
                        action='store_true',
                        help="Whether to use peakCSE model")
    parser.add_argument("--use-flat-nep-prior", 
                        default=False,
                        action='store_true',
                        help="Whether to use flat prior on NEPs")
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

    np.random.seed(args.sampling_seed) # for reproducibility
    
    NMAX_NSAT = 25
    NB_CSE = args.nb_cse

    ### NEP priors
    prior_list = []
    fixed_parameters = {}
    if not args.use_flat_nep_prior:
        print("Running with multi-variate gaussian prior on NEPs")
        # add the E_sat prior, more stable with very tight prior on it to mimic fixing it
        E_sat_prior = GaussianPrior(-16.0, 0.005, parameter_names=["E_sat"]) # extremely tight
        prior_list.append(E_sat_prior)
        # this multivariate normal distribution is the result from analyzng the
        # nuclear experimental results as shown in Tab I in arxiv:2310.11588
        mean = jnp.array([34.44274178, 85.66934584, 7.98315033, 185.35530836])
        cov = jnp.array(
            [[ 3.92995634e+00,  3.97843945e+01,  1.82178480e+02, -1.05126954e-01],
             [ 3.97843945e+01,  4.98093523e+02,  2.65119374e+03, -1.89098628e+00],
             [ 1.82178480e+02,  2.65119374e+03,  1.60024311e+04, -6.90419859e+00],
             [-1.05126954e-01, -1.89098628e+00, -6.90419859e+00,  4.32077816e+02]]
        )
        NEP_prior = MultivariateGaussianDistribution(
            mean=mean, cov=cov, parameter_names=['E_sym', 'L_sym', 'K_sym', 'K_sat']
        )
        prior_list.append(NEP_prior)
    else:
        print("Running with uniform prior on NEPs")
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

    ### CSE priors
    if args.use_peakCSE or NB_CSE > 0:
        if args.fix_nbreak:
            print(f"nbreak is fixed to {args.fix_nbreak}")
            fixed_parameters['nbreak'] = args.fix_nbreak
        else:
            nbreak_prior = UniformPrior(
                1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]
            )
            prior_list.append(nbreak_prior)
    if args.use_peakCSE:
        print(f"Using peakCSE for higer density extension")
        prior_list.extend([
            UniformPrior(0.1, 1.0, parameter_names=["gaussian_peak"]),
            UniformPrior(2 * 0.16, 12 * 0.16, parameter_names=["gaussian_mu"]),
            UniformPrior(0.1 * 0.16, 5 * 0.16, parameter_names=["gaussian_sigma"]),
            UniformPrior(0.1, 1.0, parameter_names=["logit_growth_rate"]),
            UniformPrior(2 * 0.16, 35 * 0.16, parameter_names=["logit_midpoint"]),
        ])
    elif NB_CSE > 0:
        print(f"Using {NB_CSE} speed-of-sound extension segments")
        for i in range(NB_CSE):
            prior_list.append(
                UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"])
            )
            if not args.fix_ngrid:
                # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
                prior_list.append(
                    UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"])
                )
            else:
                fixed_parameters[f"n_CSE_{i}_u"] = 0.5

        # Final point to end
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

    # non-GR modification
    assert args.GR_or_nonGR in ["GR", "nonGR"], "Unknown gravity choice"
    fixed_parameters.update({
        'lambda_BL': 0.,
        'lambda_HB': 1.,
        'alpha': 0.0,
        'beta': 0.0,
        'gamma': 0.0,
    })
    if args.GR_or_nonGR == 'nonGR':
        prior_list.append(UniformPrior(-0.5, 0.5, parameter_names=['lambda_DY']))
        #prior_list.append(UniformPrior(-0.5, 1.0, parameter_names=['gamma']))
        #if args.fix_alpha:
        #    print(f"alpha is fixed to {args.fix_alpha}")
        #    fixed_parameters['alpha'] = args.fix_alpha
        #else:
        #    prior_list.append(UniformPrior(5., 10., parameter_names=['alpha']))
        #if args.fix_beta:
        #    print(f"beta is fixed to {args.fix_beta}")
        #    fixed_parameters['beta'] = args.fix_beta
        #else:
        #    prior_list.append(UniformPrior(0.05, 0.30, parameter_names=['beta']))
    else:
        # fixing all these non-GR parameters
        #fixed_parameters['alpha'] = 0.
        #fixed_parameters['beta'] = 0.
        #fixed_parameters['gamma'] = 0.
        fixed_parameters['lambda_DY'] = 0.

    # Construct the EOS prior and a transform here which can be used down below for creating the EOS plots after inference is completed
    eos_prior = CombinePrior(prior_list)
    eos_param_names = eos_prior.parameter_names
    all_output_keys = [
        "logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS",
        "n", "p", "h", "e", "dloge_dlogp", "cs2"
    ]
    if args.GR_or_nonGR == "nonGR":
        all_output_keys.extend(["lambda_DY",])
        #all_output_keys.extend(["alpha", "beta", "gamma", "lambda_DY"])

    name_mapping = (eos_param_names, all_output_keys)
    
    # Create the output directory if it does not exist
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Copy this script to the output directory, for reproducibility later on
    shutil.copy(__file__, os.path.join(outdir, "backup_inference.py"))
    
    keep_names = ["alpha", "beta", "gamma", "lambda_DY"]
    if args.sample_GW170817:
        prior_list.append(
            UniformPrior(
                1.18, 1.20,
                parameter_names=["chirp_mass_GW170817"]
            )
        )
        prior_list.append(
            UniformPrior(
                -20., -3.,
                parameter_names=["log_shifted_eta_GW170817"]
            )
        )
        keep_names += [
            "chirp_mass_GW170817", "log_shifted_eta_GW170817"
        ]
    if args.sample_GW190425:
        prior_list.append(
            UniformPrior(
                1.36, 1.51,
                parameter_names=["chirp_mass_GW190425"]
            )
        )
        prior_list.append(
            UniformPrior(
                -12., -4.,
                parameter_names=["log_shifted_eta_GW190425"]
            )
        )
        keep_names += [
            "chirp_mass_GW190425", "log_shifted_eta_GW190425"
        ]
    if args.sample_J0030:
        prior_list.append(
            UniformPrior(1.0, 2.1, parameter_names=["mass_J0030"])
        )
        keep_names += ["mass_J0030",]
    if args.sample_J0740:
        prior_list.append(
            UniformPrior(1.8, 2.5, parameter_names=["mass_J0740"])
        )
        keep_names += ["mass_J0740",]
    if args.sample_J0437:
        prior_list.append(
            UniformPrior(1.0, 2.0, parameter_names=["mass_J0437"])
        )
        keep_names += ["mass_J0437",]
    
    ##################
    ### LIKELIHOOD ###
    ##################

    # Likelihood: choose which PSR(s) to perform inference on:
    # TODO: if using binary Love, need an extra toggle for that
    if not args.use_zero_likelihood:
        # GWs
        likelihoods_list_GW = []
        if args.sample_GW170817:
            print(f"Loading data necessary for the event GW170817")
            likelihoods_list_GW.append(
                utils.GWLikelihood_with_masses(
                    nf_path=args.GW170817_data,
                    gw_name='GW170817'
                )
            )
        if args.sample_GW190425:
            print(f"Loading data necessary for the event GW190425")
            likelihoods_list_GW.append(
                utils.GWLikelihood_with_masses(
                    nf_path=args.GW190425_data,
                    gw_name='GW190425'
                )
            )
        # NICER
        likelihoods_list_NICER = []
        if args.sample_J0030:
            print(f"Loading data necessary for the event J0030")
            likelihoods_list_NICER.append(
                utils.NICERLikelihood_with_masses(
                    nf_dict={
                        'Amsterdam': args.J0030_data_Amsterdam,
                        'Maryland': args.J0030_data_Maryland,
                    },
                    psr_name="J0030"
                )
            )
        if args.sample_J0740:
            print(f"Loading data necessary for the event J0740")
            # load data
            likelihoods_list_NICER.append(
                utils.NICERLikelihood_with_masses(
                    nf_dict={
                        'Amsterdam': args.J0740_data_Amsterdam,
                        'Maryland': args.J0740_data_Maryland,
                    },
                    psr_name="J0740"
                )
            )
        if args.sample_J0437:
            print(f"Loading data necessary for the event J0437")
            # load data
            likelihoods_list_NICER.append(
                utils.NICERLikelihood_with_masses(
                    nf_dict={
                        'Amsterdam': args.J0437_data_Amsterdam,
                        'Maryland': args.J0437_data_Maryland,
                    },
                    psr_name="J0437"
                )
            )
        # Radio timing mass measurement pulsars
        likelihoods_list_radio = []
        if args.sample_radio_J1614:
            likelihoods_list_radio.append(
                utils.RadioTimingLikelihood("J1614", 1.937, 0.014)
            )
        if args.sample_radio_J0348:
            likelihoods_list_radio.append(
                utils.RadioTimingLikelihood("J0348", 2.01, 0.04)
            )
        if args.sample_GW190814:
            likelihoods_list_radio.append(
                utils.RadioTimingLikelihood("GW190814", 2.579, 0.056)
            )
        # PREX and CREX
        likelihoods_list_REX = []
        if args.sample_PREX:
            print(f"Loading data necessary for PREX")
            likelihoods_list_REX.append(
                utils.REXLikelihood(
                    samples_dict=pd.read_csv(args.PREX_data, header=0, delimiter=' '),
                    experiment_name="PREX"
                )
            )
        if args.sample_CREX:
            print(f"Loading data necessary for CREX")
            likelihoods_list_REX.append(
                utils.REXLikelihood(
                    samples_dict=pd.read_csv(args.CREX_data, header=0, delimiter=' '),
                    experiment_name="CREX"
                )
            )
        if len(likelihoods_list_REX) == 0:
            print(f"Not sampling PREX or CREX data now")
            
        # Chiral EFT
        likelihoods_list_chiEFT = []
        # FIXME: only add chiEFT if we are sampling MM+CSE, ignore during MM-only
        if args.sample_chiEFT and args.nb_cse > 0:
            keep_names += ["nbreak"]
            print(f"Loading data necessary for the Chiral EFT")
            likelihoods_list_chiEFT += [utils.ChiEFTLikelihood()]

        # Total likelihoods list:
        likelihoods_list = \
            likelihoods_list_GW \
            + likelihoods_list_NICER \
            + likelihoods_list_radio \
            + likelihoods_list_REX \
            + likelihoods_list_chiEFT
        print(f"Sanity checking: likelihoods_list = {likelihoods_list}")
        print(f"Number of likelihood funtions used = {len(likelihoods_list)}")
        likelihood = utils.CombinedLikelihood(likelihoods_list)
        
    # Construct the transform object
    TOV_output_keys = ["masses_EOS", "radii_EOS", "Lambdas_EOS"]
    prior = CombinePrior(prior_list)
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, TOV_output_keys)
    my_transform = utils.MicroToMacroTransform(
        name_mapping,
        keep_names = None,
        nmax_nsat = NMAX_NSAT,
        nb_CSE = NB_CSE,
        peakCSE = args.use_peakCSE,
        fixed_params = fixed_parameters,
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
    p_for_logpcutoff = (log_prob > np.quantile(log_prob, 0.001)).astype(float)
    p_for_logpcutoff /= np.sum(p_for_logpcutoff)
    idx = np.random.choice(
        np.arange(len(log_prob)),
        size=args.N_samples_EOS,
        replace=False,
        p=p_for_logpcutoff,
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
    # now get back the log-likelihood
    log_likelihood = log_prob - log_prior
    max_logL = jnp.amax(log_likelihood)
    print(f"The maximum log-likelihood is: {max_logL}")
    # output the result
    np.savez(
        os.path.join(args.outdir, "eos_samples.npz"),
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
