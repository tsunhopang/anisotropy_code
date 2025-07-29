import numpy as np
import sys
sys.path.insert(0, '/data/gravwav/thopang/projects/post_tov/scripts/')
import utils
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Post-processing script for generating all additional parameters and making corner-plot")
    parser.add_argument("--input", 
                        type=str, 
                        required=True,
                        help="Which EOS file to handle.")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    return parser.parse_args()

def main(args):

    # load the data
    data = dict(np.load(args.input))
    # list of parameters to plot
    plotting_parameters = ['E_sym', 'L_sym', 'K_sym', 'K_sat']
    # convert all EOS parameters
    data = utils.EOS_conversion(data)
    plotting_parameters.extend(
        [
            'R_14',
            'TOV_mass',
        ]
    )
    # check which input are used for the analysis
    if 'chirp_mass_GW170817' in data.keys():
        # convert all parameters for this GW event
        data = utils.GW_conversion(data, 'GW170817')
        plotting_parameters.extend(
            [
                'chirp_mass_GW170817',
                'mass_ratio_GW170817',
                'lambda_tilde_GW170817',
                'delta_lambda_tilde_GW170817'
            ]
        )
    if 'chirp_mass_GW190425' in data.keys():
        # convert all parameters for this GW event
        data = utils.GW_conversion(data, 'GW190425')
        plotting_parameters.extend(
            [
                'chirp_mass_GW190425',
                'mass_ratio_GW190425',
                'lambda_tilde_GW190425',
                'delta_lambda_tilde_GW190425'
            ]
        )
    if 'mass_J0030' in data.keys():
        # convert all parameters for this NICER pulsar
        data = utils.NICER_conversion(data, 'J0030')
        plotting_parameters.extend(
            [
                'mass_J0030',
                'radius_J0030'
            ]
        )
    if 'mass_J0740' in data.keys():
        # convert all parameters for this NICER pulsar
        data = utils.NICER_conversion(data, 'J0740')
        plotting_parameters.extend(
            [
                'mass_J0740',
                'radius_J0740'
            ]
        )
    for key in ['gamma', 'beta']:
        if np.var(data[key]) != 0.:
            plotting_parameters.append(key)

    plotting_parameters.append('log_likelihood')

    # drop the M-R-Lambda to save space
    data.pop('masses_EOS')
    data.pop('radii_EOS')
    data.pop('Lambdas_EOS')
    np.savez(f'{args.outdir}/eos_samples_complete.npz', **data)
    #utils.plot_corner(args.outdir, data, plotting_parameters)

if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
