# The Dynamics of Task Interactions in Deep Linear Multi-Task Networks
Code for paper "The Dynamics of Task Interactions in Deep Linear Multi-Task Networks" published in the Transactions on Machine Learning Research. 

The goal of this paper is to provide insight into how task alignment (defined in terms of singular vectors) shapes the way multiple tasks interact in linear multi-task networks. The repository contains all the code to reproduce the results presented in the paper. Exact reproduction of the main papers figures is possible by running the correpsonding notebooks. 

<img width="928" height="370" alt="image" src="https://github.com/user-attachments/assets/6b8ccd22-4353-40b0-b2e8-405147008570" />
Figure 3. **Column A**. _Analytical solutions (black dotted lines) match numerical simulations of task-specific losses (task~1: green; task~2: blue solid lines) and_, **Columns B-C**, _the components of the network functions for a linear neural network. The network is trained on fully aligned (top row), conflicting (middle row), and orthogonal (bottom row).  _
**Columns D-E**. Corresponding cosine similarity between task-specific gradients (GradSim) and gradient magnitudes (GradMag) for both tasks over training until convergence._ 


## Repository contents

- `Figure3_TMLR.ipynb`, `Figure4TMLR.ipynb`, and `Figure5_TMLR.ipynb` are notebooks that can be run to immediately generate the corresponding figures from the paper. `Dynamicweighting.ipynb` contains additional related analyses from the appendix.
- `networks/` contains the numerical network models and utilities used for the training experiments.
- `qqt/` contains the analytical simulation code.
- `run_cluster_experiment.py`, `example_submit_Multi_MNIST.sh`, and `example_submit_cpu_nyuv2_linear.slurm` provide example cluster runs for the Multi-MNIST and NYUv2 experiments.
- `Experiment Logs/` contains logs and saved outputs from the experiments used to produce the specific plots, including Multi-MNIST and NYUv2 results.

The remaining Python files provide shared dataset, training, logging, task-overlap, and utility functionality used by the notebooks and experiments.


## Documentation
For the exact configuration of each high-dimensional experiment please see the TMLR paper.
