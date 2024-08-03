from __future__ import absolute_import, print_function

import os
import argparse
from collections import OrderedDict

from rfsd.distributions import rff_cauchy_sampler
from rfsd.goftest import RFDGofTest
from rfsd.kernel import KGauss2
from rfsd.util import (Timer, store_objects, restore_object,
                       create_folder_if_not_exist, meddistance)
from rfsd import rfsd

from kgof import kernel, density, goftest
from kgof.util import fit_gaussian_draw
from nystroem_ksd.goftest import NystroemKSD

from rfsd.experiments import config

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds', type=int, default=10)
    parser.add_argument('-o', '--output-dir', default='results')
    parser.add_argument('-c', '--factor-of-nys-points', default=4, type=int)
    parser.add_argument('-d', '--dimension', default=10, type=int)
    return parser.parse_args()


def main():
    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})
    args = parse_arguments()
    output_dir = args.output_dir

    create_folder_if_not_exist(output_dir)
    os.chdir(output_dir)
    print('changed working directory to', output_dir)

    ns = [100, 500, 1000, 2000, 5000]
    d = args.dimension
    J = 10
    reps = args.rounds

    p = density.IsotropicNormal(np.zeros(d), 1)
    def make_divergence_call(k):
        return lambda dat: k.divergence(dat.data(), J=J)

    def fssd(dat):
        V = fit_gaussian_draw(dat.data(), J, seed=4, reg=1e-6)
        return goftest.GaussFSSD(p, 1, V).compute_stat(dat)
    
    def fssd_opt(dat):
        med_l2 = meddistance(dat.data(), subsample=1000)
        sigma2 = med_l2**2

        tr, te = dat.split_tr_te(tr_proportion=0.2, seed=None)
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand)
        kinit = kernel.KGauss(sigma2*2)
        list_gwidth = np.hstack( ( (sigma2)*gwidth_factors ) )
        Vgauss = fit_gaussian_draw(tr.data(), J, reg=1e-6,
                               seed=int(10*np.sum(tr.data()**2)))
        V0 = Vgauss
        besti, objs = goftest.GaussFSSD.grid_search_gwidth(p, tr, V0,
                                                       list_gwidth)
        gwidth = list_gwidth[besti]

        ops = {
            'reg': 1e-2,
            'max_iter': 40,
            'tol_fun': 1e-4,
            'disp': False,
            'locs_bounds_frac': 10.0,
            'gwidth_lb': 1e-1,
            'gwidth_ub': 1e4,
            }
        Vgauss_opt, gwidth_opt, info = goftest.GaussFSSD.optimize_locs_widths(p, tr,
                gwidth, V0, **ops)
        
        return goftest.GaussFSSD(p, gwidth_opt, Vgauss_opt, seed=4).compute_stat(te)
    
    def rff_gauss(dat):
        med_l2 = meddistance(dat.data(), subsample=1000)
        sigma2 = med_l2**2
        kgauss = KGauss2(sigma2)
        rff = rfsd.RFFKSD(kgauss.rff_sampler(d), p)
        return RFDGofTest(p, rff).compute_stat(dat.data())

    def rff_cauchy(dat):
        med_l2 = meddistance(dat.data(), subsample=1000)
        rff = rfsd.RFFKSD(rff_cauchy_sampler(med_l2, d), p)
        return  RFDGofTest(p, rff).compute_stat(dat.data())

    metrics = OrderedDict([
        #('Gauss FSSD-rand', fssd),
        #('Gauss FSSD-opt', fssd_opt),
        #('IMQ KSD', make_divergence_call(rfsd.KSD(kernel.KIMQ(), p))),
        #('Gauss KSD', make_divergence_call(rfsd.KSD(kernel.KGauss(1), p))),
        #('L2 SechExp', make_divergence_call(rfsd.LrSechFastKSD(p))),
        #('L1 IMQ', make_divergence_call(rfsd.L1IMQFastKSD(p, d=d))),
        #('Gauss RFF', rff_gauss),
        #('Cauchy RFF', rff_cauchy),
        ('Nys IMQ KSD', make_divergence_call(NystroemKSD(p, kernel.KIMQ(), m=lambda n : args.factor_of_nys_points* int(np.sqrt(n))))),
        ('Nys Gauss KSD', make_divergence_call(NystroemKSD(p, kernel.KGauss(1), m=lambda n : args.factor_of_nys_points* int(np.sqrt(n))))),
        ])

    store_loc = 'speed-experiment-stored-data-%d-rounds' % reps

    try:
        times = restore_object(store_loc, 'times')
        print('reloaded existing data')
    except IOError:
        times = OrderedDict([(k, np.zeros((len(ns), reps))) for k in metrics])
        for i, n in enumerate(ns):
            print('n =', n)
            for j in range(reps):
                dat = p.get_datasource().sample(n, seed=None)
                for kname, f in metrics.items():
                    with Timer() as t:
                        f(dat)
                    times[kname][i,j] = t.interval
        store_objects(store_loc, times=times)

    # plot the results
    base_fig_name = 'speed-experiment-%d-rounds' % reps
    color_dict = config.test_name_colors_dict()

    plt.figure()
    plt.clf()
    for kname, ktimes in times.items():
        ts = np.mean(ktimes, axis=1)
        plt.plot(ns, ts, label=kname, c=color_dict[kname])
    plt.xlabel('sample size $N$')
    plt.yscale('log')
    plt.ylabel('time (seconds)')
    sns.despine()
    plt.savefig(base_fig_name + '-no-legend.png', bbox_inches='tight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)
    plt.savefig(base_fig_name + '.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
