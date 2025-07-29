import numpy as np
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
    data = utils.Psnm_conversion(data, densities=[0.32,], labels=['2nsat',])
    plotting_parameters.append('P_snm_2nsat')

    data = utils.Esym_fixed_density_conversion(
            data, densities=[0.05,], labels=['alphaD',]
            )
    plotting_parameters.append('E_sym_alphaD')

    for exp in ['Mass_Skyrme', 'Mass_DFT', 'IAS', 'HIC_Isodiff', 'HIC_npratio', 'HIC_pi']:
        data = utils.Esym_conversion(data, exp)
        plotting_parameters.append(f'E_sym_{exp}')

    data = utils.Psym_conversion(data, 'HIC_pi')
    plotting_parameters.append('P_sym_HIC_pi')

    data = utils.Psym_fixed_density_conversion(data, [0.24,], ['HIC_npflow',])
    plotting_parameters.append('P_sym_HIC_npflow')

    np.savez(f'{args.outdir}/eos_samples_complete.npz', **data)
    utils.plot_corner(args.outdir, data, plotting_parameters)

if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
