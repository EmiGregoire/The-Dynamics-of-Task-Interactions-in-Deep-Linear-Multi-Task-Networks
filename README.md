# The-Dynamics-of-Task-Interactions-in-Deep-Linear-Multi-Task-Networks
Code for paper "The Dynamics of Task Interactions in Deep Linear Multi-Task Networks" published in TMLR. 

## Repository contents

- `Figure3_TMLR.ipynb`, `Figure4TMLR.ipynb`, and `Figure5_TMLR.ipynb` are notebooks that can be run to immediately generate the corresponding figures from the paper. `Dynamicweighting.ipynb` contains additional related analyses from the appendix.
- `networks/` contains the numerical network models and utilities used for the training experiments.
- `qqt/` contains the analytical simulation code.
- `run_cluster_experiment.py`, `example_submit_Multi_MNIST.sh`, and `example_submit_cpu_nyuv2_linear.slurm` provide example cluster runs for the Multi-MNIST and NYUv2 experiments.
- `Experiment Logs/` contains logs and saved outputs from the experiments used to produce the specific plots, including Multi-MNIST and NYUv2 results.

The remaining Python files provide shared dataset, training, logging, task-overlap, and utility functionality used by the notebooks and experiments.


## Documentation
For the exact configuration of each high-dimensional experiment please see the TMLR paper.
