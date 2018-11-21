import argparse
import contextlib
import time

import matplotlib.pyplot as plt
import numpy as np
import six
import scipy.io as sio

import cupy as cp


@contextlib.contextmanager
def timer(message):
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def sample(p):
    p_cpu = cp.asnumpy(p)
    index = np.random.rand() <= (np.cumsum(p_cpu)/np.sum(p_cpu))
    index = np.where(index)
    index = np.min(index)
    index_gpu = cp.asarray(index)
    return index_gpu


def msmgenerate(nframe, T, emission, pi):
    xp = cp.get_array_module(cp.asarray(T))
    state = xp.zeros(nframe, dtype=np.int32)
    data = xp.zeros(nframe, dtype=np.int32)
    state[0] = sample(pi)
    data[0] = sample(emission[state[0], :])
    for iframe in xp.arange(1, nframe):
        state[iframe] = sample(T[state[iframe-1], :])
        data[iframe] = sample(emission[state[iframe-1], :])
    return state, data


def msmforward(data_list, T, emission, pi):
    xp = cp.get_array_module(cp.asarray(T))
    ndata = len(data_list)
    nstate = T[0, :].size
    logL = xp.zeros(ndata, dtype=np.float64)
    alpha_list = []
    factor_list = []
    for idata in range(ndata):
        data = data_list[idata]
        nframe = data.size
        alpha = xp.zeros((nframe, nstate), dtype=np.float64)
        factor = xp.zeros(nframe, dtype=np.float64)
        alpha[0, :] = pi*emission[:, data[0]]
        factor[0] = xp.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :]/factor[0]
        for iframe in range(1, nframe):
            alpha[iframe, :] = xp.sum(alpha[iframe-1, :, None] * T, axis=0) * emission[:, data[iframe]]
            factor[iframe] = xp.sum(alpha[iframe, :])
            alpha[iframe, :] = alpha[iframe, :]/factor[iframe]
        logL[idata] = xp.sum(xp.log(factor))
        alpha_list.append(alpha)
        #print(alpha)
        factor_list.append(factor)
    return logL, alpha_list, factor_list


def msmbackward(data_list, factor_list, T, emission, pi):
    xp = cp.get_array_module(cp.asarray(T))
    ndata = len(data_list)
    nstate = T[0, :].size
    logL = xp.zeros(ndata, dtype=np.float64)
    beta_list = []
    for idata in range(ndata):
        data = data_list[idata]
        factor = factor_list[idata]
        nframe = data.size
        beta = xp.zeros((nframe, nstate), dtype=np.float64)
        beta[-1, :] = 1.0;
        for iframe in range(nframe-2, -1, -1):
            d = emission[:, data[iframe+1]] * beta[iframe+1, :]
            beta[iframe, :] = xp.sum(T * d, axis=1) / factor[iframe+1]
        logL[idata] = xp.sum(xp.log(factor))
        beta_list.append(beta)
    return logL, beta_list


def msmbaumwelch(data_list, T0, emission0, pi0):
    xp = cp.get_array_module(cp.asarray(T0))
    ## setup
    TOLERANCE = 10**(-4)
    check_convergence = float('inf')
    count_iteration = 0
    logL_old = 1.0
    if not isinstance(data_list, list):
        data_list = [data_list]
    ndata = len(data_list)
    nobs = emission0[0, :].size
    nstate = T0[0, :].size
    while check_convergence > TOLERANCE:
        ## E-step
        with timer(' E-step forward'):
            logL, alpha_list, factor_list = msmforward(data_list, T0, emission0, pi0);
        with timer(' E-step backward'):
            logL2, beta_list = msmbackward(data_list, factor_list, T0, emission0, pi0);
        with timer(' E-step map'):
            log_alpha_list = list(map(xp.log, alpha_list))
            log_beta_list = list(map(xp.log, beta_list))
            log_T0 = xp.log(T0)
            log_emission0 = xp.log(emission0)
        ## M-step
        # pi
        # pi = xp.zeros(nstate, dtype=np.float64)
        # log_gamma_list = []
        # for idata in range(ndata):
        #     log_gamma_list.append(log_alpha_list[idata] + log_beta_list[idata])
        #     pi = pi + xp.exp(log_gamma_list[idata][0, :])
        # pi = pi/xp.sum(pi)
        pi = pi0
        # emission
        # emission = xp.zeros((nstate, nobs), dtype=np.float64)
        # for idata in range(ndata):
        #     data = data_list[idata]
        #     for istate in range(nstate):
        #         for iobs in range(nobs):
        #             id = (data == iobs)
        #             if xp.any(id):
        #                 emission[istate, iobs] = emission[istate, iobs] + xp.sum(xp.exp(log_gamma_list[idata][id, istate]))
        # emission[xp.isnan(emission)] = 0.0
        # emission = emission / xp.sum(emission, axis=1)[:, None]
        emission = emission0
        # T
        with timer(' M-step'):
            T = xp.zeros((nstate, nstate), dtype=np.float64)
            for idata in range(ndata):
                data = data_list[idata]
                nframe = data.size
                for iframe in range(1, nframe):
                    log_xi = log_alpha_list[idata][iframe-1, :, None] + log_beta_list[idata][iframe, :]
                    T = T + xp.exp(log_xi + log_emission0[:, data[iframe]] + log_T0)/factor_list[idata][iframe]
            T[xp.isnan(T)] = 0.0
            T = T / xp.sum(T, axis=1)[:, None]
        ## Check convergence
        count_iteration += 1
        logL = xp.sum(logL)
        check_convergence = xp.abs(logL_old - logL)
        print('%d iteration LogLikelihood = %e  delta = %e  tolerance = %e' % (count_iteration, logL, check_convergence, TOLERANCE))
        logL_old = logL
        pi0 = pi
        emission0 = emission
        T0 = T
    return T, emission, pi


def run(gpuid, n_clusters, num, max_iter, use_custom_kernel, output):
    xp = cp.get_array_module(cp.asarray([1.0]))
    T = xp.array([[0.1, 0.7, 0.2],
                  [0.2, 0.1, 0.7],
                  [0.7, 0.2, 0.1]], dtype=np.float64)
    emission = xp.array([[0.9, 0.1],
                         [0.6, 0.4],
                         [0.1, 0.9]], dtype=np.float64)
    pi = xp.array([8.0, 1.0, 1.0], dtype=np.float64)
    nframe = 1000
    state, data_list = msmgenerate(nframe, T, emission, pi)
    T0 = xp.array([[0.1, 0.1, 0.8],
                  [0.8, 0.1, 0.1],
                  [0.3, 0.2, 0.5]], dtype=np.float64)
    emission0 = xp.array([[0.9, 0.1],
                         [0.6, 0.4],
                         [0.1, 0.9]], dtype=np.float64)
    pi0 = xp.array([0.8, 0.1, 0.1], dtype=np.float64)

    # matlab = sio.loadmat('test_baumwelch.mat')
    # data = xp.asarray(matlab['data'], dtype=xp.int).flatten() - 1
    # T = xp.asarray(matlab['T'], dtype=np.float64)
    # emission = xp.asarray(matlab['emission'], dtype=np.float64)
    # pi = xp.asarray(matlab['pi_i'], dtype=np.float64).flatten()
    # T0 = xp.asarray(matlab['T0'], dtype=np.float64)
    # data_list = [data]

    with timer('Baum-Welch'):
        T1, emission1, pi1 = msmbaumwelch(data_list, T0, emission, pi);
    print(T)
    print(T0)
    print(T1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument('--n-clusters', '-n', default=2, type=int,
                        help='number of clusters')
    parser.add_argument('--num', default=5000000, type=int,
                        help='number of samples')
    parser.add_argument('--max-iter', '-m', default=10, type=int,
                        help='number of iterations')
    parser.add_argument('--use-custom-kernel', action='store_true',
                        default=False, help='use Elementwise kernel')
    parser.add_argument('--output-image', '-o', default=None, type=str,
                        help='output image file name')
    args = parser.parse_args()
    run(args.gpu_id, args.n_clusters, args.num, args.max_iter,
        args.use_custom_kernel, args.output_image)
