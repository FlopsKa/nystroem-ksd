#!/bin/sh

python scripts/run_gof_experiment.py laplace
python scripts/run_gof_experiment.py student-t
python scripts/run_gof_experiment.py rbm
python scripts/run_gof_experiment.py null
python scripts/run_speed_experiment.py