import numpy as np

import kgof.data as data
import kgof.kernel as kernel
import kgof.density as density
from kgof.goftest import KernelSteinTest

import nystroem_ksd.goftest as goftest

def test_h_p_Gauss_kernel():
    """The h_p method constructs the correct Gram matrix with the Gauss kernel."""
    n = 100
    m = 10
    d = 10
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(size=(n,d))
    k = kernel.KGauss(sigma2=1/2)
    p = density.IsotropicNormal(0, 1)
    _, ref = KernelSteinTest(p=p,k=k).compute_stat(data.Data(X),return_ustat_gram=True)

    # test m=n
    actual = goftest.NystroemKSD(p=p, k=k).h_p(X=X, Y=X)
    np.testing.assert_allclose(ref,actual)

    # test m!=n
    Y = X[[*range(m)]]
    actual = goftest.NystroemKSD(p=p, k=k).h_p(X=X, Y=Y)
    np.testing.assert_allclose(ref[:n,:m],actual)

def test_h_p_IMQ_kernel():
    """The h_p method constructs the correct Gram matrix with the IMQ kernel."""
    n = 100
    m = 10
    d = 10
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(size=(n,d))
    k = kernel.KIMQ()
    p = density.IsotropicNormal(0, 1)
    _, ref = KernelSteinTest(p=p,k=k).compute_stat(data.Data(X),return_ustat_gram=True)

    # test m=n
    actual = goftest.NystroemKSD(p=p, k=k).h_p(X=X, Y=X)
    np.testing.assert_allclose(ref,actual)

    # test m!=n by selecting a subset as reference
    Y = X[[*range(m)]]
    actual = goftest.NystroemKSD(p=p, k=k).h_p(X=X, Y=Y)
    np.testing.assert_allclose(ref[:n,:m],actual)

def test_Nystroem_based_statistic_similar_to_quad_time_statistic():
    n = 2000
    d = 1
    rng = np.random.default_rng(1234)
    X = data.Data(rng.standard_normal(size=(n,d)))
    k = kernel.KGauss(sigma2=1/2)
    p = density.IsotropicNormal(0, 1)
    ref_stat = KernelSteinTest(p=p,k=k).compute_stat(X,return_ustat_gram=False)

    actual_stat = goftest.NystroemKSD(p=p, k=k).compute_stat(X)
    np.testing.assert_almost_equal(ref_stat/n, actual_stat,decimal=5)

def test_Nystroem_based_H0_holds():
    n = 2000
    d = 1
    rng = np.random.default_rng(1234)
    X = data.Data(rng.standard_normal(size=(n,d)))
    k = kernel.KGauss(sigma2=1/2)
    p = density.IsotropicNormal(0, 1)

    results = goftest.NystroemKSD(p=p, k=k).perform_test(X)
    assert results["h0_rejected"] == False

def test_Nystroem_based_H1_holds():
    n = 2000
    d = 1
    rng = np.random.default_rng(1234)
    X = data.Data(rng.normal(loc=2,size=(n,d)))
    k = kernel.KGauss(sigma2=1/2)
    p = density.IsotropicNormal(0, 1)

    results = goftest.NystroemKSD(p=p, k=k).perform_test(X)
    assert results["h0_rejected"] == True
