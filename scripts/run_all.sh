#!/bin/sh

python scripts/run_gof_experiment.py laplace -o full_results
python scripts/run_gof_experiment.py student-t -o full_results
python scripts/run_gof_experiment.py rbm -o full_results
python scripts/run_speed_experiment.py -o full_results