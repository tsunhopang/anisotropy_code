import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, logit
from jaxtyping import Float
from jax.scipy.stats import gaussian_kde, norm
import pandas as pd
import copy

from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform
from jimgw.prior import UniformPrior, CombinePrior

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from jesterTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family_nonGR
import jesterTOV.utils as jester_utils

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

class MicroToBulkTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = [],
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 nmax_nsat: float = 3,
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

        self.ns = jnp.linspace(0., self.nmax, num=self.ndat_metamodel)
        self.xs = (self.ns - 0.16) / 3. / 0.16
        
        #self.eos_pnm = MetaModel_EOS_model(
        #    nmax_nsat = self.nmax_nsat,
        #    ndat = self.ndat_metamodel,
        #    proton_fraction=1e-3
        #    )
        #self.eos_snm = MetaModel_EOS_model(
        #    nmax_nsat = self.nmax_nsat,
        #    ndat = self.ndat_metamodel,
        #    proton_fraction=0.5
        #    )
        self.transform_func = self.transform_func_MM
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params 
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
                
        print("Fixed params loaded inside the MicroToBulkTransform:")
        for key, value in self.fixed_params.items():
            print(f"    {key}: {value}")
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}

        ns = self.ns
        xs = self.xs
        es_snm = params['E_sat'] + params['K_sat'] * xs * xs / 2.
        es_sym = params['E_sym'] + params['L_sym'] * xs + params['K_sym'] * xs * xs / 2.
        ps_snm = params['K_sat'] * xs * ns * ns / 3. / 0.16
        ps_sym = ns * ns / 3. / 0.16 * (params['L_sym'] + params['K_sym'] * xs)
        
        #ns, ps_pnm, es_pnm, _, _, _, _ = self.eos_pnm.construct_eos(NEP)
        #ns, ps_snm, es_snm, _, _, _, _ = self.eos_snm.construct_eos(NEP)
        ## convert all these back to usual unit
        #ns /= jester_utils.fm_inv3_to_geometric
        #ps_pnm /= jester_utils.MeV_fm_inv3_to_geometric
        #ps_snm /= jester_utils.MeV_fm_inv3_to_geometric
        #es_pnm /= jester_utils.MeV_fm_inv3_to_geometric
        #es_snm /= jester_utils.MeV_fm_inv3_to_geometric
        return_dict = {
            "n_EOS": ns, "Esym_EOS": es_sym,
            "Psym_EOS": ps_sym, "Psnm_EOS": ps_snm
        }

        return return_dict
        
        
###################
### LIKELIHOODS ###
###################
class EsymLikelihood_with_densities(LikelihoodBase):
    
    def __init__(
        self,
        mu: float,
        sigma: float,
        exp_name: str,
        transform: MicroToBulkTransform = None):
        
        self.mu = mu
        self.sigma = sigma
        self.exp_name = exp_name
        self.transform = transform

    def __repr__(self):
        return f"{self.__class__.__name__} with mu={self.mu} and sigma={self.sigma}"
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        n_EOS, Esym_EOS = params["n_EOS"], params["Esym_EOS"]
        den = params[f"den_{self.exp_name}"]
        Esym = jnp.interp(jnp.array([den]), n_EOS, Esym_EOS, left=0, right=0)
        
        log_likelihood = norm.logpdf(Esym, loc=self.mu, scale=self.sigma)
        
        return log_likelihood.at[0].get()


class EsymLikelihood_without_densities(LikelihoodBase):
    
    def __init__(
        self,
        mu: float,
        sigma: float,
        exp_name: str,
        exp_den: float,
        transform: MicroToBulkTransform = None):
        
        self.mu = mu
        self.sigma = sigma
        self.exp_name = exp_name
        self.exp_den = jnp.array([exp_den])
        self.transform = transform

    def __repr__(self):
        return f"{self.__class__.__name__} with mu={self.mu} and sigma={self.sigma} at density {self.exp_den}"
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        n_EOS, Esym_EOS = params["n_EOS"], params["Esym_EOS"]
        Esym = jnp.interp(self.exp_den, n_EOS, Esym_EOS, left=0, right=0)
        
        log_likelihood = norm.logpdf(Esym, loc=self.mu, scale=self.sigma)
        
        return log_likelihood.at[0].get()


class PsymLikelihood_with_densities(LikelihoodBase):
    
    def __init__(
        self,
        mu: float,
        sigma: float,
        exp_name: str,
        transform: MicroToBulkTransform = None):
        
        self.mu = mu
        self.sigma = sigma
        self.exp_name = exp_name
        self.transform = transform

    def __repr__(self):
        return f"{self.__class__.__name__} with mu={self.mu} and sigma={self.sigma}"
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        n_EOS, Psym_EOS = params["n_EOS"], params["Psym_EOS"]
        den = params[f"den_{self.exp_name}"]
        Psym = jnp.interp(jnp.array([den]), n_EOS, Psym_EOS, left=0, right=0)
        
        log_likelihood = norm.logpdf(Psym, loc=self.mu, scale=self.sigma)
        
        return log_likelihood.at[0].get()

class PsymLikelihood_without_densities(LikelihoodBase):
    
    def __init__(
        self,
        mu: float,
        sigma: float,
        exp_name: str,
        exp_den: float,
        transform: MicroToBulkTransform = None):
        
        self.mu = mu
        self.sigma = sigma
        self.exp_name = exp_name
        self.exp_den = jnp.array([exp_den])
        self.transform = transform

    def __repr__(self):
        return f"{self.__class__.__name__} with mu={self.mu} and sigma={self.sigma} at density {self.exp_den}"
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        n_EOS, Psym_EOS = params["n_EOS"], params["Psym_EOS"]
        Psym = jnp.interp(self.exp_den, n_EOS, Psym_EOS, left=0, right=0)
        
        log_likelihood = norm.logpdf(Psym, loc=self.mu, scale=self.sigma)
        
        return log_likelihood.at[0].get()


class PsnmLikelihood_without_densities(LikelihoodBase):
    
    def __init__(
        self,
        mu: float,
        sigma: float,
        exp_name: str,
        exp_den: float,
        transform: MicroToBulkTransform = None):
        
        self.mu = mu
        self.sigma = sigma
        self.exp_name = exp_name
        self.exp_den = jnp.array([exp_den])
        self.transform = transform

    def __repr__(self):
        return f"{self.__class__.__name__} with mu={self.mu} and sigma={self.sigma} at density {self.exp_den}"
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        n_EOS, Psnm_EOS = params["n_EOS"], params["Psnm_EOS"]
        Psnm = jnp.interp(self.exp_den, n_EOS, Psnm_EOS, left=0, right=0)
        
        log_likelihood = norm.logpdf(Psnm, loc=self.mu, scale=self.sigma)
        
        return log_likelihood.at[0].get()
        
    
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
                 transform: MicroToBulkTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.transform = transform
        self.counter = 0
        if self.transform:
            self.fixed_params = self.transform.fixed_params
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        params.update(self.fixed_params)
        return 0.0


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
        # NEPs
        'E_sym': r'$E_{\rm{sym}} \ [\rm{MeV}]$',
        'L_sym': r'$L_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sym': r'$K_{\rm{sym}} \ [\rm{MeV}]$',
        'K_sat': r'$K_{\rm{sat}} \ [\rm{MeV}]$',
        # Psnm
        'P_snm_2nsat': r'$P_{\rm{snm}}(2n_{\rm{sat}}) \ [\rm{MeV fm}^{-3}]$',
        # Esym
        'E_sym_alphaD': r'$E_{\rm{sym}}(\alpha_{D}) \ [\rm{MeV}]$',
        'E_sym_Mass_Skyrme': r'$E_{\rm{sym}}(\rm{Skyrme}) \ [\rm{MeV}]$',
        'E_sym_Mass_DFT': r'$E_{\rm{sym}}(\rm{DFT}) \ [\rm{MeV}]$',
        'E_sym_IAS': r'$E_{\rm{sym}}(\rm{IAS}) \ [\rm{MeV}]$',
        'E_sym_HIC_Isodiff': r'$E_{\rm{sym}}(\rm{Isodiff}) \ [\rm{MeV}]$',
        'E_sym_HIC_npratio': r'$E_{\rm{sym}}(\rm{n/p \ ratio}) \ [\rm{MeV}]$',
        'E_sym_HIC_pi': r'$E_{\rm{sym}}(\pi) \ [\rm{MeV}]$',
        # Psym
        'P_sym_HIC_pi': r'$P_{\rm{sym}}(\pi) \ [\rm{MeV}]$',
        'P_sym_HIC_npflow': r'$P_{\rm{sym}}(\rm{n/p \ flow}) \ [\rm{MeV fm}^{-3}]$',
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


def Psnm_conversion(data, densities=[0.32,], labels=['2nsat']):

    n_EOS = data['n_EOS']
    Esym_EOS = data['Esym_EOS']
    Psym_EOS = data['Psym_EOS']
    Psnm_EOS = data['Psnm_EOS']

    N_samples = n_EOS.shape[0]

    output_array = []
    for i in tqdm(range(N_samples)):
        output_array.append(np.interp(densities, n_EOS[i], Psnm_EOS[i]))
    output_array = np.array(output_array).T

    for i in range(len(labels)):
        data[f'P_snm_{labels[i]}'] = output_array[i]

    return data


def Esym_fixed_density_conversion(data, densities=[0.05,], labels=['alphaD']):

    n_EOS = data['n_EOS']
    Esym_EOS = data['Esym_EOS']
    Psym_EOS = data['Psym_EOS']
    Psnm_EOS = data['Psnm_EOS']

    N_samples = n_EOS.shape[0]

    output_array = []
    for i in tqdm(range(N_samples)):
        output_array.append(np.interp(densities, n_EOS[i], Esym_EOS[i]))
    output_array = np.array(output_array).T

    for i in range(len(labels)):
        data[f'E_sym_{labels[i]}'] = output_array[i]

    return data


def Esym_conversion(data, exp_name):

    n_EOS = data['n_EOS']
    Esym_EOS = data['Esym_EOS']
    Psym_EOS = data['Psym_EOS']
    Psnm_EOS = data['Psnm_EOS']

    N_samples = n_EOS.shape[0]

    output_array = []
    for i in tqdm(range(N_samples)):
        output_array.append(np.interp(data[f'den_{exp_name}'][i], n_EOS[i], Esym_EOS[i]))

    data[f'E_sym_{exp_name}'] = np.array(output_array)

    return data


def Psym_fixed_density_conversion(data, densities=[0.05,], labels=['HIC_npflow']):

    n_EOS = data['n_EOS']
    Esym_EOS = data['Esym_EOS']
    Psym_EOS = data['Psym_EOS']
    Psnm_EOS = data['Psnm_EOS']

    N_samples = n_EOS.shape[0]

    output_array = []
    for i in tqdm(range(N_samples)):
        output_array.append(np.interp(densities, n_EOS[i], Psym_EOS[i]))
    output_array = np.array(output_array).T

    for i in range(len(labels)):
        data[f'P_sym_{labels[i]}'] = output_array[i]

    return data


def Psym_conversion(data, exp_name):

    n_EOS = data['n_EOS']
    Esym_EOS = data['Esym_EOS']
    Psym_EOS = data['Psym_EOS']
    Psnm_EOS = data['Psnm_EOS']

    N_samples = n_EOS.shape[0]

    output_array = []
    for i in tqdm(range(N_samples)):
        output_array.append(np.interp(data[f'den_{exp_name}'][i], n_EOS[i], Psym_EOS[i]))

    data[f'P_sym_{exp_name}'] = np.array(output_array)

    return data
