import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import equinox as eqx
from flowjax.train import fit_to_data
from flowjax.distributions import Normal, Uniform, Transformed
from flowjax.bijections import RationalQuadraticSpline, Affine, Invert
from flowjax.flows import block_neural_autoregressive_flow, masked_autoregressive_flow
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import corner
import copy
import utils
params = {
    "axes.grid": True,
    "text.usetex": False,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "figure.titlesize": 16
}
plt.rcParams.update(params)

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.95, 0.997],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        density=True,
                        save=False)

def make_cornerplot(chains_1, 
                    chains_2,
                    parameter_range: list[float],
                    name: str):
    """
    Plot a cornerplot of the true data samples and the NF samples
    Note: the shape use is a bit inconsistent below, watch out.
    """

    plt.figure(np.random.randint(1000))
    # The training data:
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_1d_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    fig = corner.corner(chains_1, range=parameter_range, **corner_kwargs)

    # The data from the normalizing flow
    corner_kwargs["color"] = "red"
    hist_1d_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    corner.corner(chains_2, range=parameter_range, fig = fig, **corner_kwargs)

    # Make a textbox for labeling
    fs = 32
    plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

    plt.savefig(name, bbox_inches = "tight")

################
### Argparse ###
################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Normalizing flow traning on the given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", 
                        type=str, 
                        required=True,
                        help="Path to the input npz dataset")
    parser.add_argument("--label", 
                        type=str, 
                        required=True,
                        help="Label of the trained model")
    parser.add_argument("--output", 
                        type=str, 
                        required=True,
                        help="Path to the output trained model")
    parser.add_argument("--parameters", 
                        type=str, 
                        default="E_sat,K_sat,E_sym,L_sym,K_sym,lambda_DY",
                        help="Comma seperated list of parameters to train on")
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="Seed used for training")
    parser.add_argument("--Nsamples",
                        type=utils.int_or_none,
                        default=None,
                        help="Number of samples used for training")
    return parser.parse_args()

def main(args):

    # prepare the keys
    flow_key, train_key, sample_key = jax.random.split(jax.random.key(args.seed), 3)
    # load the data
    if args.input.endswith('npz'):
        data = np.load(args.input)
    elif args.input.endswith(('.dat', '.txt', '.csv')):
        data = pd.read_csv(args.input, header=0, delimiter=' ')
    # prepare the data into expected shape
    parameters = args.parameters.split(',')
    print(f"Training on the parameters: {parameters}")
    n_dim = len(parameters)
    x = []
    for parameter in parameters:
        x.append(np.array(data[parameter]))
    x = np.array(x).T
    if args.Nsamples:
        print(f'Using the first {args.Nsamples} samples')
        x = x[:args.Nsamples]
    else:
        print(f'Using all the samples provided')
    # standardize the input samples
    preprocess = Affine(-x.mean(axis=0) / x.std(axis=0), 1. / x.std(axis=0))
    x_processed = jax.vmap(preprocess.transform)(x)
    # create the flow model
    flow = masked_autoregressive_flow(
        key=flow_key,
        base_dist=Normal(jnp.zeros(x.shape[1])),
        transformer=RationalQuadraticSpline(knots=10, interval=5),
    )
    flow, loss = fit_to_data(
        key=train_key,
        dist=flow,
        x=x_processed,
        learning_rate=5e-4,
        max_epochs=1000,
        max_patience=30,
    )
    flow = Transformed(flow, Invert(preprocess))
    # output the model and reload for validation
    print("Saving the normalizing flow")
    model_path = f'{args.output}/{args.label}_NF.eqx'
    eqx.tree_serialise_leaves(model_path, flow)
    # define clean structures
    clean_flow = Transformed(
        masked_autoregressive_flow(
            key=jax.random.key(42),
            base_dist=Normal(jnp.zeros(x.shape[1])),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
        ),
        Invert(Affine(jnp.zeros(x.shape[1]), jnp.ones(x.shape[1])))
    )
    # loading the flow
    loaded_flow = eqx.tree_deserialise_leaves(model_path, clean_flow)
    # sample from the flow for checking
    nf_samples_reloaded = np.array(loaded_flow.sample(sample_key, (x.shape[0], )))
    # fetch the range
    parameter_range=np.array([
        [
            np.min(x.T[i]),
            np.max(x.T[i])
        ]
        for i in range(n_dim)
    ])
    # make corner plot for visualization
    make_cornerplot(
        x, nf_samples_reloaded,
        parameter_range=parameter_range,
        name=f'{args.output}/{args.label}_reloaded_corner.png'
    )

if __name__ == "__main__":
    args = parse_arguments()  # Get command-line arguments
    main(args)
