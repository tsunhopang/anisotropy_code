# Code for measuring pressure anisotropy

This repository contains code used for performing Bayesian inference of pressure anisotropy among neutron star observations (see [paper](https://arxiv.org/abs/2507.13039)) It is organized for clarity and reproducibility, as part of a submission for peer review.
The code includes
- normalizing flow training
- hierarchical Bayesian inference
- joint Bayesian infrence
- evidence estimation
- postprocessing tools

---

## 📁 Repository Structure
```
.
├── nfs/ # Pretrained normalizing flow models
│ ├── astro/ # (If applicable) Astrophysical observations
│ ├── metamodel/ # NF models on posterior using metamodel
│ └── metamodel_peak/ # NF models on posterior using metamodel+peak
│
├── nuclear_scripts/ # Estimation for prior on NEPs
│ ├── inference.py # Inference pipeline
│ ├── postprocessing.py # Posterior and diagnostic analysis
│ ├── utils.py # Shared utility functions
│ └── NEP_cov_estimation.py # NEPs covariance estimation
│
├── scripts/ # Core computational scripts
│ ├── comparison.py
│ ├── evidence.py # Bayesian evidence estimation
│ ├── hierarchical_inference.py # Hierarchical Bayesian analysis
│ ├── hierarchical_inference_no_Gaussian.py # Hierarchical Bayesian analysis without Gaussian hyper-distribution
│ ├── inference.py # Joint Bayesian analysis
│ ├── nf_training.py # Script for normalizing flow training
│ ├── postprocessing.py # Postprocessing
│ └── utils.py # Shared utility functions
│
└── README.md # Project description and instructions
```

## 🔧 Dependencies

This project requires the following Python packages:

- [jax](https://github.com/jax-ml/jax) 
- [jester](https://github.com/nuclear-multimessenger-astronomy/jester)
- [harmonic](https://github.com/astro-informatics/harmonic)
- [flowjax](https://github.com/danielward27/flowjax)
