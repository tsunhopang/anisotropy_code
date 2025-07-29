import numpy as np
from tqdm import tqdm
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp, logit
from jaxtyping import Float
from jax.scipy.stats import gaussian_kde, norm
import pandas as pd
import copy

from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform
from jimgw.prior import Prior, UniformPrior, CombinePrior

import equinox as eqx
from flowjax.distributions import Normal, Transformed 
from flowjax.bijections import Affine, Invert, RationalQuadraticSpline
from flowjax.flows import masked_autoregressive_flow

from jesterTOV.eos import MetaModel_with_peakCSE_EOS_model, MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family_nonGR
import jesterTOV.utils as jose_utils

#################
### CONSTANTS ###
#################

NEP_CONSTANTS_DICT = {
    # This is a set of MM parameters that gives a decent initial guess for Hauke's Set A maximum likelihood EOS
    "E_sym": 33.431808,
    "L_sym": 77.178344,
    "K_sym": -129.761344,
    "Q_sym": 422.442807,
    "Z_sym": -1644.011429,
    
    "E_sat": -16.0,
    "K_sat": 285.527411,
    "Q_sat": 652.366343,
    "Z_sat": -1290.138303,
    
    "nbreak": 0.153406,
    
    # FIXME: this has been changed now because of uniform [0, 1] sampling!
    # "n_CSE_0": 3 * 0.16,
    # "n_CSE_1": 4 * 0.16,
    # "n_CSE_2": 5 * 0.16,
    # "n_CSE_3": 6 * 0.16,
    # "n_CSE_4": 7 * 0.16,
    # "n_CSE_5": 8 * 0.16,
    # "n_CSE_6": 9 * 0.16,
    # "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5,
    "cs2_CSE_1": 0.7,
    "cs2_CSE_2": 0.5,
    "cs2_CSE_3": 0.4,
    "cs2_CSE_4": 0.8,
    "cs2_CSE_5": 0.6,
    "cs2_CSE_6": 0.9,
    "cs2_CSE_7": 0.8,
    
    # This is the final entry
    "cs2_CSE_8": 0.9,
}

def merge_dicts(dict1: dict, dict2: dict):
    """
    Merges 2 dicts, but if the key is already in dict1, it will not be overwritten by dict2.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary. Do not use its values if keys are in dict1
    """
    
    result = {}
    for key, value in dict1.items():
        result[key] = value
        
    for key, value in dict2.items():
        if key not in result.keys():
            result[key] = value
            
    return result


##################
### TRANSFORMS ###
##################

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = [],
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 peakCSE: bool = False,
                 # TOV kwargs
                 min_nsat_TOV: float = 0.75,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                 fixed_params: dict[str, float] = {},
                ):
    
        # By default, keep all names
        if keep_names is None:
            keep_names = name_mapping[0]
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.peakCSE = peakCSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Create the EOS object -- there are several choices for the parametrizations
        if peakCSE:
            print("Using MetaModel + peakCSE model")
            eos = MetaModel_with_peakCSE_EOS_model(
                nmax_nsat=self.nmax_nsat,
                ndat_metamodel=self.ndat_metamodel,
                ndat_CSE=self.ndat_CSE,
            )
            self.transform_func = self.transform_func_MM_peakCSE
        elif nb_CSE > 0:
            print("Using MetaModel + CSE model")
            eos = MetaModel_with_CSE_EOS_model(
                nmax_nsat=self.nmax_nsat,
                ndat_metamodel=self.ndat_metamodel,
                ndat_CSE=self.ndat_CSE,
            )
            self.transform_func = self.transform_func_MM_CSE
        else:
            print("Using MetaModel model")
            eos = MetaModel_EOS_model(
                nmax_nsat = self.nmax_nsat,
                ndat = self.ndat_metamodel
            )
            self.transform_func = self.transform_func_MM
        
        self.eos = eos
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params 
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
                
        print("Fixed params loaded inside the MicroToMacroTransform:")
        for key, value in self.fixed_params.items():
            print(f"    {key}: {value}")
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family_nonGR(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)
        eos_tuple = (
            ns, ps, hs, es, dloge_dlogps,
            params['alpha'], params['beta'], params['gamma'],
            params['lambda_BL'], params['lambda_DY'], params['lambda_HB']
            )
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {
            "logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS,
            "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
            "n": ns, "p": ps, "h": hs, "e": es,
            "dloge_dlogp": dloge_dlogps, "cs2": cs2,}

        return return_dict

    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        width = (self.nmax - params["nbreak"]) / self.nb_CSE

        ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
        ngrids = params["nbreak"] + jnp.cumsum(ngrids_u) * width
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE + 1)])
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (
            ns, ps, hs, es, dloge_dlogps,
            params['alpha'], params['beta'], params['gamma'],
            params['lambda_BL'], params['lambda_DY'], params['lambda_HB']
            )
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {
            "logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS,
            "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
            "n": ns, "p": ps, "h": hs, "e": es,
            "dloge_dlogp": dloge_dlogps, "cs2": cs2,}
        
        return return_dict

    def transform_func_MM_peakCSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and peakCSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        peakCSE = {key: value for key, value in params.items() if (("_sat" not in key) and ("_sym" not in key))}
        peakCSE.pop("nbreak")
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, peakCSE)
        eos_tuple = (
            ns, ps, hs, es, dloge_dlogps,
            params['alpha'], params['beta'], params['gamma'],
            params['lambda_BL'], params['lambda_DY'], params['lambda_HB']
            )
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {
            "logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS,
            "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
            "n": ns, "p": ps, "h": hs, "e": es,
            "dloge_dlogp": dloge_dlogps, "cs2": cs2,}
        
        return return_dict

    
class ChirpMassMassRatioToSourceComponentMasses(NtoMTransform):
        
    def __init__(
        self,
    ):
        name_mapping = (["M_c", "q", "d_L"], ["m_1", "m_2"])
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.transform_func = detector_frame_M_c_q_to_source_frame_m_1_m_2
        
class ChirpMassMassRatioToLambdas(NtoMTransform):
    
    def __init__(
        self,
        name_mapping,
    ):
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.mass_transform = ChirpMassMassRatioToSourceComponentMasses()
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        masses_EOS = params["masses_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        
        # Get masses
        m_params = self.mass_transform.forward(params)
        m_1, m_2 = m_params["m_1"], m_params["m_2"]
        
        # Interpolate to get Lambdas
        lambda_1_interp = jnp.interp(m_1, masses_EOS, Lambdas_EOS, right = -1.0)
        lambda_2_interp = jnp.interp(m_2, masses_EOS, Lambdas_EOS, right = -1.0)
        
        return {"lambda_1": lambda_1_interp, "lambda_2": lambda_2_interp}
        
        
###################
### LIKELIHOODS ###
###################
class NICERLikelihood_with_masses(LikelihoodBase):
    
    def __init__(
            self,
            nf_dict: dict,
            psr_name: str,
            transform: MicroToMacroTransform = None
        ):
        
        self.psr_name = psr_name
        self.transform = transform

        # define a clean flow struct for loading the NFs
        n_dim = 2
        clean_flow = Transformed(
            masked_autoregressive_flow(
                key=jax.random.key(42),
                base_dist=Normal(jnp.zeros(n_dim)),
                transformer=RationalQuadraticSpline(
                    knots=10, interval=5
                ),
            ),
            Invert(Affine(jnp.zeros(n_dim), jnp.ones(n_dim)))
        )
        # Load the NFs
        self.amsterdam_posterior = eqx.tree_deserialise_leaves(
            nf_dict["Amsterdam"], clean_flow
        )
        self.maryland_posterior = eqx.tree_deserialise_leaves(
            nf_dict["Maryland"], clean_flow
        )
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS, radii_EOS = params["masses_EOS"], params["radii_EOS"]
        mass = params[f"mass_{self.psr_name}"]
        radius = jnp.interp(mass, masses_EOS, radii_EOS, left=0, right=0)
        
        mr_grid = jnp.vstack([mass, radius]).T
        logL_amsterdam = self.amsterdam_posterior.log_prob(mr_grid)
        logL_maryland = self.maryland_posterior.log_prob(mr_grid)
        
        logL_array = jnp.array([logL_maryland, logL_amsterdam])
        log_likelihood = logsumexp(logL_array, axis=0) - jnp.log(2)
        
        return log_likelihood.at[0].get()
    
    
class GWLikelihood_with_masses(LikelihoodBase):

    def __init__(
            self,
            nf_path: str, 
            gw_name: str,
            transform: MicroToMacroTransform = None,
            very_negative_value: float = -9999999.0
        ):
        
        self.transform = transform
        self.gw_name = gw_name
        self.counter = 0
        self.very_negative_value = very_negative_value

        # define a clean flow struct for loading the NFs
        n_dim = 4
        clean_flow = Transformed(
            masked_autoregressive_flow(
                key=jax.random.key(42),
                base_dist=Normal(jnp.zeros(n_dim)),
                transformer=RationalQuadraticSpline(
                    knots=10, interval=5
                ),
            ),
            Invert(Affine(jnp.zeros(n_dim), jnp.ones(n_dim)))
        )
        # Load the NFs
        self.NS_posterior = eqx.tree_deserialise_leaves(
            nf_path, clean_flow
        )

    def LI_lambdaTdlambdaT_giveneta_logprior(self, eta):
        a = (8. / 13.) * (1. + 7. * eta - 31. * eta * eta)
        b = (8. / 13.) * jnp.sqrt(1. - 4. * eta) * (1. + 9. * eta - 11. * eta * eta)
        c = (1. / 2.) * jnp.sqrt(1. - 4. * eta) * (1. - 13272. * eta / 1319. + 8944. * eta * eta / 1319.)
        d = (1. / 2.) * (1. - 15910. * eta / 1319. + 32850. * eta * eta / 1319. + 3380. * eta * eta * eta / 1319.)

        prior = 1. / (b * c - a * d)
  
        return jnp.log(prior)
        

    def evaluate(self, params: dict[str, float], data: dict) -> float:

        # fetch the parameters
        Mc = params[f"chirp_mass_{self.gw_name}"]
        log_shifted_eta = params[f"log_shifted_eta_{self.gw_name}"]
        # convert into the usual space
        eta = 0.25 - jnp.exp(log_shifted_eta)
        Mtot = Mc / jnp.power(eta, 0.6)
        tmp = 1. / eta / 2. - 1.
        q = tmp - jnp.sqrt(tmp * tmp - 1)
        m1 = Mtot / (1. + q)
        m2 = Mtot * q/ (1. + q)
        # check if the masses exceed the TOV mass 
        masses_EOS, Lambdas_EOS = params['masses_EOS'], params['Lambdas_EOS']
        mtov = jnp.max(masses_EOS)
        penalty_mass1_mtov = jnp.where(m1 > mtov, self.very_negative_value, 0.0)
        penalty_mass2_mtov = jnp.where(m2 > mtov, self.very_negative_value, 0.0)

        # Lambdas: interpolate to get the values
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right = 1.0)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right = 1.0)

        # convert the samples to the kde's space
        lambda_plus = lambda_1 + lambda_2
        lambda_minus = lambda_1 - lambda_2
        lambdaT = 8. / 13. * (
            (1. + 7. * eta - 31. * eta**2) * lambda_plus +
            jnp.sqrt(1. - 4. * eta) * (1. + 9. * eta - 11. * eta**2) * lambda_minus)
        dlambdaT = 1. / 2. * (
            jnp.sqrt(1 - 4 * eta) * (1. - 13272. / 1319. * eta + 8944. / 1319. * eta**2) *
            lambda_plus + (1. - 15910. / 1319. * eta + 32850. / 1319. * eta**2 +
                           3380. / 1319. * eta**3) * lambda_minus)

        # Make a 4D array of the m1, m2, and lambda values and evalaute log prob on it
        ml_grid = jnp.array([
            Mc, log_shifted_eta, lambdaT, dlambdaT
        ]).T
        logpdf_NS = self.NS_posterior.log_prob(ml_grid)
        # correct for the prior on lambdas
        logpdf_NS -= self.LI_lambdaTdlambdaT_giveneta_logprior(eta)
        
        log_likelihood = logpdf_NS + penalty_mass1_mtov + penalty_mass2_mtov
        
        return log_likelihood

    
class REXLikelihood(LikelihoodBase):
    
    def __init__(self,
                 samples_dict: dict,
                 experiment_name: str):
        
        self.experiment_name = experiment_name
        self.counter = 0
        
        # Load the data
        self.posterior =  gaussian_kde(
            jnp.vstack([jnp.array(samples_dict['Esym']), jnp.array(samples_dict['Lsym'])])
        )
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood_array = self.posterior.logpdf(
            jnp.array([params["E_sym"], params["L_sym"]])
        )
        log_likelihood = log_likelihood_array.at[0].get()

        return log_likelihood
    
class RadioTimingLikelihood(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 mean: float, 
                 std: float,
                 transform: MicroToMacroTransform = None):
        
        self.psr_name = psr_name
        self.transform = transform
        
        self.mean = mean
        self.std = std
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        masses_EOS = params["masses_EOS"]
        mtov = jnp.max(masses_EOS)
        log_likelihood = norm.logcdf(
            mtov, loc=self.mean, scale=self.std
        )
        log_likelihood -= jnp.log(mtov)
        
        return log_likelihood

    
class ChiEFTLikelihood(LikelihoodBase):
    
    def __init__(self,
                 transform: MicroToMacroTransform = None,
                 nb_n: int = 100):
        
        self.transform = transform
        
        # Load the chi EFT data
        low_filename = "/home/twouters2/projects/jax_tov_eos/paper_jose/src/paper_jose/inference/data/chiEFT/low.dat"
        f = np.loadtxt(low_filename)
        n_low = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_low = jnp.array(f[:, 1])
        # NOTE: this is not a spline but it is the best I can do -- does this matter? Need to check later on
        EFT_low = lambda x: jnp.interp(x, n_low, p_low)
        
        high_filename = "/home/twouters2/projects/jax_tov_eos/paper_jose/src/paper_jose/inference/data/chiEFT/high.dat"
        f = np.loadtxt(high_filename)
        n_high = jnp.array(f[:, 0]) / 0.16 # convert to nsat
        p_high = jnp.array(f[:, 1])
        
        EFT_high = lambda x: jnp.interp(x, n_high, p_high)
        
        self.n_low = n_low
        self.p_low = p_low
        self.EFT_low = EFT_low
        
        self.n_high = n_high
        self.p_high = p_high
        self.EFT_high = EFT_high
        
        self.nb_n = nb_n
        
        # TODO: remove once debugged
        # print(f"Init of chiEFT likelihood")
        # print("self.n_low range")
        # print(jnp.min(self.n_low))
        # print(jnp.max(self.n_low))
        
        # print("self.n_high")
        # print(jnp.min(self.n_high))
        # print(jnp.max(self.n_high))
        
        # print("self.p_low range")
        # print(jnp.min(self.p_low))
        # print(jnp.max(self.p_low))
        
        # print("self.p_high")
        # print(jnp.min(self.p_high))
        # print(jnp.max(self.p_high))
        
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # Get relevant parameters
        n, p = params["n"], params["p"]
        nbreak = params["nbreak"]
        
        # Convert to nsat for convenience
        nbreak = nbreak / 0.16
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        
        # TODO: remove once debugged
        # print("nbreak:")
        # print(nbreak)
        
        # print("n_range:")
        # print(jnp.min(n))
        # print(jnp.max(n))
        
        # print("p_range:")
        # print(jnp.min(p))
        # print(jnp.max(p))
        
        prefactor = 1 / (nbreak - 0.75 * 0.16)
        
        # Lower limit is at 0.12 fm-3
        this_n_array = jnp.linspace(0.75, nbreak, self.nb_n)
        dn = this_n_array.at[1].get() - this_n_array.at[0].get()
        low_p = self.EFT_low(this_n_array)
        high_p = self.EFT_high(this_n_array)
        
        # Evaluate the sampled p(n) at the given n
        sample_p = jnp.interp(this_n_array, n, p)
        
        # Compute f
        def f(sample_p, low_p, high_p):
            beta = 6/(high_p-low_p)
            return_value = (
                -beta * (sample_p - high_p) * jnp.heaviside(sample_p - high_p, 0) +
                -beta * (low_p - sample_p) * jnp.heaviside(low_p - sample_p, 0) +
                1 * jnp.heaviside(sample_p - low_p, 0) * jnp.heaviside(high_p - sample_p, 0) # FIXME: 0 or 1? Hauke has 1 but then low_p with the log
            )
            
            return return_value
            
        f_array = f(sample_p, low_p, high_p) # Well actually already log f
        
        log_likelihood = prefactor * jnp.sum(f_array) * dn
        
        return log_likelihood
        
    
class CombinedLikelihood(LikelihoodBase):
    
    def __init__(self,
                 likelihoods_list: list[LikelihoodBase],
                 ):
        
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.counter = 0
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        all_log_likelihoods = jnp.array([likelihood.evaluate(params, data) for likelihood in self.likelihoods_list])
        return jnp.sum(all_log_likelihoods)
    
class ZeroLikelihood(LikelihoodBase):
    def __init__(self,
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.transform = transform
        self.counter = 0
        if self.transform:
            self.fixed_params = self.transform.fixed_params
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_params)
        return 0.0


class HierarchicalLikelihood(LikelihoodBase):
    
    def __init__(self,
                 nf_list: list,
                 EOS_parameters: list = ["E_sat", "K_sat", "E_sym", "L_sym", "K_sym"],
                 gamma_bound: list = [-0.5, 0.5],
                 ):
        
        self.nf_list = nf_list
        self.n_ns = len(self.nf_list)
        self.EOS_parameters = EOS_parameters
        self.lower_bound = gamma_bound[0]
        self.upper_bound = gamma_bound[1]

    def convert_u_to_gamma(self, u, mu, sigma):
        lower_cdf = norm.cdf(self.lower_bound, loc=mu, scale=sigma)
        upper_cdf = norm.cdf(self.upper_bound, loc=mu, scale=sigma)
        return norm.ppf(u * (upper_cdf - lower_cdf) + lower_cdf, loc=mu, scale=sigma)
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # first calculate the prior correction
        # do it in post-processing is more stable
        log_prior = (self.n_ns - 1.) * jnp.log(1. / (self.upper_bound - self.lower_bound))
        # convert gamma_u to gamma
        gamma = self.convert_u_to_gamma(
            jnp.array([params[f'gamma_u_{i}'] for i in range(self.n_ns)]),
            params['gamma_mu'], params['gamma_sigma']
        )
        EOS_params = jnp.stack([params[k] for k in self.EOS_parameters]).T
        logpdf = []
        for i, nf in enumerate(self.nf_list):
            # combine EOS_params (N, 5) with gamma[i] (N,) to make input (N, 6)
            inputs = jnp.concatenate([EOS_params, gamma[i].reshape(-1)])
            # evaluate log_prob for all N samples, shape of (N,)
            logpdf.append(self.nf_list[i].log_prob(inputs))
        # vmap over all the NFs
        logpdf = jnp.array(logpdf).sum(axis=0)
        # correct it by unweighting the prior
        logpdf -= log_prior

        return logpdf

class HierarchicalLikelihoodNoGaussian(LikelihoodBase):
    
    def __init__(self,
                 nf_list: list,
                 EOS_parameters: list = ["E_sat", "K_sat", "E_sym", "L_sym", "K_sym"],
                 gamma_bound: list = [-0.5, 0.5],
                 ):
        
        self.nf_list = nf_list
        self.n_ns = len(self.nf_list)
        self.EOS_parameters = EOS_parameters
        self.lower_bound = gamma_bound[0]
        self.upper_bound = gamma_bound[1]

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # first calculate the prior correction
        # do it in post-processing is more stable
        log_prior = (self.n_ns - 1.) * jnp.log(1. / (self.upper_bound - self.lower_bound))
        gamma = jnp.array([params[f'gamma_{i}'] for i in range(self.n_ns)])
        EOS_params = jnp.stack([params[k] for k in self.EOS_parameters]).T
        logpdf = []
        for i, nf in enumerate(self.nf_list):
            # combine EOS_params (N, 5) with gamma[i] (N,) to make input (N, 6)
            inputs = jnp.concatenate([EOS_params, gamma[i].reshape(-1)])
            # evaluate log_prob for all N samples, shape of (N,)
            logpdf.append(self.nf_list[i].log_prob(inputs))
        # vmap over all the NFs
        logpdf = jnp.array(logpdf).sum(axis=0)
        # correct it by unweighting the prior
        logpdf -= log_prior

        return logpdf


def compute_log_prior_for_i(i, result, prior):
    y = {key: value[i] for key, value in result.items()}
    return prior.log_prob(y)
# Vectorize the function with vmap
vectorized_log_prior = jax.vmap(compute_log_prior_for_i, in_axes=(0, None, None))


def plot_corner(outdir, samples, parameters, truths=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    matplotlib.rcParams.update({
        'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})

    kwargs = dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), title_quantiles=[0.16, 0.5, 0.84],
        show_titles=True, color='#0072C1',
        truth_color='tab:orange', quantiles=[0.05, 0.95],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, truths=truths, hist_kwargs={'density': True})

    labels_dict = {
        'log_likelihood': r'$\log\mathcal{L}$',
        'E_sym': r'$E_{\rm{sym}} \ [\rm{MeV}]$',
        'L_sym': r'$L_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sym': r'$K_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sat': r'$K_{\rm{sat}} \ [\rm{MeV}]$',
        'R_14': r'$R_{1.4} \ [\rm{km}]$',
        'alpha': r'$\alpha$',
        'beta': r'$C^*$',
        'lambda_DY': r'$\gamma_0$',
        'gamma': r'$\gamma_1$',
        'TOV_mass': r'$M_{\rm{max}} \ [M_{\odot}]$',
        'mass_1_GW170817': r'$m_{1,{\rm GW170817}} \ [M_{\odot}]$',
        'mass_2_GW170817': r'$m_{2,{\rm GW170817}} \ [M_{\odot}]$',
        'chirp_mass_GW170817': r'$\mathcal{M}_{\rm GW170817} \ [M_{\odot}]$',
        'symmetric_mass_ratio_GW170817': r'$\eta_{\rm GW170817}$',
        'mass_ratio_GW170817': r'$q_{\rm GW170817}$',
        'log_shifted_eta_GW170817': r'$\log\Delta\eta_{\rm GW170817}$',
        'lambda_1_GW170817': r'$\Lambda_1_{\rm GW170817}$',
        'lambda_2_GW170817': r'$\Lambda_2_{\rm GW170817}$',
        'lambda_tilde_GW170817': r'$\tilde{\Lambda}_{\rm GW170817}$',
        'delta_lambda_tilde_GW170817': r'$\delta\tilde{\Lambda}_{\rm GW170817}$',
        'mass_1_GW190425': r'$m_{1,{\rm GW190425}} \ [M_{\odot}]$',
        'mass_2_GW190425': r'$m_{2,{\rm GW190425}} \ [M_{\odot}]$',
        'chirp_mass_GW190425': r'$\mathcal{M}_{\rm GW190425} \ [M_{\odot}]$',
        'symmetric_mass_ratio_GW190425': r'$\eta_{\rm GW190425}$',
        'mass_ratio_GW190425': r'$q_{\rm GW190425}$',
        'log_shifted_eta_GW190425': r'$\log\Delta\eta_{\rm GW190425}$',
        'lambda_1_GW190425': r'$\Lambda_1_{\rm GW190425}$',
        'lambda_2_GW190425': r'$\Lambda_2_{\rm GW190425}$',
        'lambda_tilde_GW190425': r'$\tilde{\Lambda}_{\rm GW190425}$',
        'delta_lambda_tilde_GW190425': r'$\delta\tilde{\Lambda}_{\rm GW190425}$',
        'mass_J0740': r'$M_{\rm J0740+6220} \ [M_{\odot}]$',
        'radius_J0740': r'$R_{\rm J0740+6220} \ [\rm{km}]$',
        'mass_J0030': r'$M_{\rm J0030+0451} \ [M_{\odot}]$',
        'radius_J0030': r'$R_{\rm J0030+0451} \ [\rm{km}]$',
        'mass_J0437': r'$M_{\rm J0437-4715} \ [M_{\odot}]$',
        'radius_J0437': r'$R_{\rm J0437-4715} \ [\rm{km}]$',
        'mass_J1231−': r'$M_{\rm J1231-1411} \ [M_{\odot}]$',
        'radius_J1231−': r'$R_{\rm J1231−1411} \ [\rm{km}]$',
    }

    labels = []
    plotting_samples = []
    for parameter in parameters:
        labels.append(labels_dict[parameter])
        plotting_samples.append(samples[parameter])

    plotting_samples = np.array(plotting_samples).T

    fig_index = np.random.randint(1, 1000)
    plt.figure(fig_index)
    corner.corner(plotting_samples, labels=labels, **kwargs)
    plt.savefig('{0}/corner.pdf'.format(outdir))

    return

def plot_corner_compare(
    outdir,
    samples_1, samples_2,
    label_1, label_2,
    parameters, truths=None
    ):
    import matplotlib
    from matplotlib.offsetbox import AnchoredText
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    matplotlib.rcParams.update({
        'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})

    kwargs = dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.05, 0.95],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
        plot_density=False, plot_datapoints=False, fill_contours=False,
        max_n_ticks=3, truths=truths, hist_kwargs={'density': True})

    labels_dict = {
        'E_sym': r'$E_{\rm{sym}} \ [\rm{MeV}]$',
        'L_sym': r'$L_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sym': r'$K_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sat': r'$K_{\rm{sat}} \ [\rm{MeV}]$',
        'R_14': r'$R_{1.4} \ [\rm{km}]$',
        'alpha': r'$\alpha$',
        'beta': r'$\beta$',
        'gamma': r'$\gamma$',
        'TOV_mass': r'$M_{\rm{max}} \ [M_{\odot}]$',
        'mass_1_GW170817': r'$m_{1,{\rm GW170817}} \ [M_{\odot}]$',
        'mass_2_GW170817': r'$m_{2,{\rm GW170817}} \ [M_{\odot}]$',
        'chirp_mass_GW170817': r'$\mathcal{M}_{\rm GW170817} \ [M_{\odot}]$',
        'symmetric_mass_ratio_GW170817': r'$\eta_{\rm GW170817}$',
        'mass_ratio_GW170817': r'$q_{\rm GW170817}$',
        'log_shifted_eta_GW170817': r'$\log\Delta\eta_{\rm GW170817}$',
        'lambda_1_GW170817': r'$\Lambda_1_{\rm GW170817}$',
        'lambda_2_GW170817': r'$\Lambda_2_{\rm GW170817}$',
        'lambda_tilde_GW170817': r'$\tilde{\Lambda}_{\rm GW170817}$',
        'delta_lambda_tilde_GW170817': r'$\delta\tilde{\Lambda}_{\rm GW170817}$',
        'mass_1_GW190425': r'$m_{1,{\rm GW190425}} \ [M_{\odot}]$',
        'mass_2_GW190425': r'$m_{2,{\rm GW190425}} \ [M_{\odot}]$',
        'chirp_mass_GW190425': r'$\mathcal{M}_{\rm GW190425} \ [M_{\odot}]$',
        'symmetric_mass_ratio_GW190425': r'$\eta_{\rm GW190425}$',
        'mass_ratio_GW190425': r'$q_{\rm GW190425}$',
        'log_shifted_eta_GW190425': r'$\log\Delta\eta_{\rm GW190425}$',
        'lambda_1_GW190425': r'$\Lambda_1_{\rm GW190425}$',
        'lambda_2_GW190425': r'$\Lambda_2_{\rm GW190425}$',
        'lambda_tilde_GW190425': r'$\tilde{\Lambda}_{\rm GW190425}$',
        'delta_lambda_tilde_GW190425': r'$\delta\tilde{\Lambda}_{\rm GW190425}$',
        'mass_J0740': r'$M_{\rm J0740+6220} \ [M_{\odot}]$',
        'radius_J0740': r'$R_{\rm J0740+6220} \ [\rm{km}]$',
        'mass_J0030': r'$M_{\rm J0030+0451} \ [M_{\odot}]$',
        'radius_J0030': r'$R_{\rm J0030+0451} \ [\rm{km}]$',
    }

    labels = []
    plotting_samples_1 = []
    plotting_samples_2 = []

    for parameter in parameters:
        labels.append(labels_dict[parameter])
        plotting_samples_1.append(samples_1[parameter])
        plotting_samples_2.append(samples_2[parameter])

    plotting_samples_1 = np.array(plotting_samples_1).T
    plotting_samples_2 = np.array(plotting_samples_2).T

    kwargs_1 = copy.deepcopy(kwargs)
    kwargs_2 = copy.deepcopy(kwargs)
    kwargs_2['color'] = 'C1'

    combined_samples = np.vstack([plotting_samples_1, plotting_samples_2])
    hist_range = [
        [combined_samples[:, i].min(), combined_samples[:, i].max()] \
        for i in range(combined_samples.shape[1])
    ]

    fig = corner.corner(plotting_samples_1, labels=labels, **kwargs_1)
    corner.corner(plotting_samples_2, labels=labels, fig=fig, **kwargs_2)
    title = AnchoredText("Comparison of ", loc="upper center", frameon=False, prop=dict(size=16))
    fig.add_artist(title)
    blue_text = AnchoredText(label_1, loc="upper center", frameon=False, prop=dict(size=16, color=kwargs_1['color']))
    fig.add_artist(blue_text)
    orange_text = AnchoorangeText(f" and {label_2} Distributions", loc="upper center", frameon=False, prop=dict(size=16, color=kwargs_2['color']))
    fig.add_artist(orange_text)
    plt.savefig('{0}/comparison_corner.pdf'.format(outdir))


def EOS_conversion(data):

    Lambdas_EOS = data['Lambdas_EOS']
    masses_EOS = data['masses_EOS']
    radii_EOS = data['radii_EOS']

    N_samples = Lambdas_EOS.shape[0]

    R14 = []
    Lambda14 = []
    MTOV = []

    for i in tqdm(range(N_samples)):
        R14.append(np.interp(1.4, masses_EOS[i], radii_EOS[i]))
        Lambda14.append(np.interp(1.4, masses_EOS[i], Lambdas_EOS[i]))
        MTOV.append(masses_EOS[i][-1])

    data['TOV_mass'] = np.array(MTOV)
    data['R_14'] = np.array(R14)
    data['Lambda_14'] = np.array(Lambda14)

    return data


def GW_conversion(data, name):
    Mc = data[f'chirp_mass_{name}']
    log_shifted_eta = data[f'log_shifted_eta_{name}']
    Lambdas_EOS = data['Lambdas_EOS']
    masses_EOS = data['masses_EOS']
    N_samples = len(Mc)
    # calulate all the masses
    eta = 0.25 - np.exp(log_shifted_eta)
    Mtot = Mc / np.power(eta, 0.6)
    tmp = 1. / eta / 2. - 1.
    q = tmp - np.sqrt(tmp * tmp - 1)
    m1 = Mtot / (1. + q)
    m2 = Mtot * q/ (1. + q)
    # calculate the lambdas
    lambda_1 = []
    lambda_2 = []
    for i in tqdm(range(N_samples)):
        lambda_1.append(np.interp(m1[i], masses_EOS[i], Lambdas_EOS[i]))
        lambda_2.append(np.interp(m2[i], masses_EOS[i], Lambdas_EOS[i]))
    lambda_1 = np.array(lambda_1)
    lambda_2 = np.array(lambda_2)
    # get lambdaT and dlambdaT
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    lambdaT = 8. / 13. * (
        (1. + 7. * eta - 31. * eta**2) * lambda_plus +
        np.sqrt(1. - 4. * eta) * (1. + 9. * eta - 11. * eta**2) * lambda_minus)
    dlambdaT = 1. / 2. * (
        np.sqrt(1 - 4 * eta) * (1. - 13272. / 1319. * eta + 8944. / 1319. * eta**2) *
        lambda_plus + (1. - 15910. / 1319. * eta + 32850. / 1319. * eta**2 +
                       3380. / 1319. * eta**3) * lambda_minus)
    # store the data
    data[f'mass_ratio_{name}'] = q
    data[f'mass_1_{name}'] = m1
    data[f'mass_2_{name}'] = m2
    data[f'total_mass_{name}'] = Mtot
    data[f'lambda_1_{name}'] = lambda_1
    data[f'lambda_2_{name}'] = lambda_2
    data[f'lambda_tilde_{name}'] = lambdaT
    data[f'delta_lambda_tilde_{name}'] = dlambdaT

    return data


def NICER_conversion(data, name):
    m = data[f'mass_{name}']
    masses_EOS = data['masses_EOS']
    radii_EOS = data['radii_EOS']
    N_samples = len(m)
    r = []
    for i in tqdm(range(N_samples)):
        r.append(np.interp(m[i], masses_EOS[i], radii_EOS[i]))
    r = np.array(r)

    data[f'radius_{name}'] = r

    return data


def int_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return int(value)
