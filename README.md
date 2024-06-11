# Nyström Kernel Stein Discrepancy

This is the supplement for the submission "Nyström Kernel Stein Discrepancy" and contains the code used to reproduce the experimental results.

Our code depends on the code from (Jitkrittum et. al, 2017), available [here](https://github.com/wittawatj/kernel-gof), and that of (Huggins and Mackey, 2018), available [here](https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/src/master/). For easy reference, we include both in the `lib/` directory. We updated the latter to work with recent Python packages.


## Reproducing Experiments 

To reproduce the results of the experiments, please execute the following code (requires `python>=3.11`).

    cd nystroem_ksd/
    conda create -n ksd python=3.11
    conda activate ksd
    pip install ./lib/kernel-gof/
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install ./lib/random-feature-stein-discrepancies/
    pip install -e .

    sh scripts/run_all.sh

The `notebooks/` folder contains the notebooks used for processing the results and for plotting.


## References 

Wittawat Jitkrittum, Wenkai Xu, Zoltan Szabo, Kenji Fukumizu, Arthur Gretton. A Linear-Time Kernel Goodness-of-Fit Test. NIPS, 2017.

Jonathan H. Huggins, Lester Mackey. Random Feature Stein Discrepancies. NIPS, 2018.
