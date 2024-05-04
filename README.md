# Nyström Kernel Stein Discrepancy

## Experiments 

    cd nystroem_ksd/
    conda create -n ksd python=3.11
    conda activate ksd
    pip install ./lib/kernel-gof/
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install ./lib/random-feature-stein-discrepancies/
    pip install -e .

    sh scripts/run_all.sh
