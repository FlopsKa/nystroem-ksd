# configuration as given on page 6 of http://proceedings.mlr.press/v119/grathwohl20a/grathwohl20a.pdf
python ../../scripts/run_gof_experiment.py rbm -dx 50 -dh 40 -t "Nys Gauss KSD" "Gauss KSD" -o ./50-40/
python ../../scripts/run_gof_experiment.py rbm -dx 100 -dh 80 -t "Nys Gauss KSD" "Gauss KSD" -o ./100-80/
python ../..scripts/run_gof_experiment.py rbm -dx 200 -dh 100 -t "Nys Gauss KSD" "Gauss KSD" -o ./200-100/
