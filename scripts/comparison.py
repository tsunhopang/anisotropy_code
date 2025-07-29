import numpy as np
import copy
import argparse

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
    #title = AnchoredText("Comparison of ", loc="upper center", frameon=False, prop=dict(size=16))
    #fig.add_artist(title)
    #blue_text = AnchoredText(label_1, loc="upper center", frameon=False, prop=dict(size=16, color=kwargs_1['color']))
    #fig.add_artist(blue_text)
    #orange_text = AnchoorangeText(f" and {label_2} Distributions", loc="upper center", frameon=False, prop=dict(size=16, color=kwargs_2['color']))
    #fig.add_artist(orange_text)
    plt.savefig('{0}/comparison_corner.pdf'.format(outdir))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Post-processing script for generating all additional parameters and making corner-plot")
    parser.add_argument("--input-1", 
                        type=str, 
                        required=True,
                        help="Which EOS-1 file to handle.")
    parser.add_argument("--input-2", 
                        type=str, 
                        required=True,
                        help="Which EOS-2 file to handle.")
    parser.add_argument("--label-1", 
                        type=str, 
                        required=True,
                        help="What is the label for  EOS-1.")
    parser.add_argument("--label-2", 
                        type=str, 
                        required=True,
                        help="What is the label for  EOS-2.")
    parser.add_argument("--parameters", 
                        type=str, 
                        help="Parameters to plot")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    return parser.parse_args()

def main(args):

    # load the data
    print("Loading the data")
    data_1 = dict(np.load(args.input_1))
    data_2 = dict(np.load(args.input_2))
    # list of parameters to plot
    print("Splitting the parameters keys")
    plotting_parameters = args.parameters.split(',')
    # make the plot
    print("Making the plot")
    plot_corner_compare(
        args.outdir, data_1, data_2, args.label_1, args.label_2, plotting_parameters)

if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
