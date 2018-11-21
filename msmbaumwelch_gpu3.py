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
    start = time.time()
    yield
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def sample(p):
    index = np.random.rand() <= (np.cumsum(p)/np.sum(p))
    index = np.where(index)
    index = np.min(index)
    return index


def msmgenerate(nframe, T, emission, pi):
    state = np.zeros(nframe, dtype=np.int)
    data = np.zeros(nframe, dtype=np.int)
    state[0] = sample(pi)
    data[0] = sample(emission[state[0], :])
    for iframe in np.arange(1, nframe):
        state[iframe] = sample(T[state[iframe-1], :])
        data[iframe] = sample(emission[state[iframe-1], :])
    return state, data

def msmforward(data_list, T, emission, pi):
    ndata = len(data_list)
    nstate = T[0, :].size
    logL = np.zeros(ndata, dtype=np.float64)
    alpha_list = []
    factor_list = []
    T_gpu = cp.asarray(T)
    emission_gpu = cp.asarray(emission)
    pi_gpu = cp.asarray(pi)
    for idata in range(ndata):
        data_gpu = cp.asarray(data_list[idata], dtype=np.int)
        nframe = data_gpu.size
        alpha_gpu = cp.zeros((nframe, nstate), dtype=np.float64)
        factor_gpu = cp.zeros(nframe, dtype=np.float64)
        alpha_gpu[0, :] = pi_gpu*emission_gpu[:, data_gpu[0]]
        factor_gpu[0] = cp.sum(alpha_gpu[0, :])
        alpha_gpu[0, :] = alpha_gpu[0, :]/factor_gpu[0]
        for iframe in range(1, nframe):
            alpha_gpu[iframe, :] = np.sum(alpha_gpu[iframe-1, :, None] * T_gpu, axis=0) * emission_gpu[:, data_gpu[iframe]]
            factor_gpu[iframe] = np.sum(alpha_gpu[iframe, :])
            alpha_gpu[iframe, :] = alpha_gpu[iframe, :]/factor_gpu[iframe]
        alpha = cp.asnumpy(alpha_gpu)
        factor = cp.asnumpy(factor_gpu)
        logL[idata] = np.sum(np.log(factor))
        alpha_list.append(alpha)
        #print(alpha)
        factor_list.append(factor)
    return logL, alpha_list, factor_list


def msmbackward(data_list, factor_list, T, emission, pi):
    ndata = len(data_list)
    nstate = T[0, :].size
    logL = np.zeros(ndata, dtype=np.float64)
    beta_list = []
    for idata in range(ndata):
        data = data_list[idata]
        factor = factor_list[idata]
        nframe = data.size
        beta = np.zeros((nframe, nstate), dtype=np.float64)
        beta[-1, :] = 1.0;
        for iframe in range(nframe-2, -1, -1):
            beta[iframe, :] = np.sum(T * (emission[:, data[iframe+1]] * beta[iframe+1, :]), axis=1) / factor[iframe+1]
        logL[idata] = np.sum(np.log(factor))
        beta_list.append(beta)
    return logL, beta_list


def msmbaumwelch(data_list, T0, emission0, pi0):
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
            log_alpha_list = list(map(np.log, alpha_list))
            log_beta_list = list(map(np.log, beta_list))
            log_T0 = np.log(T0)
            log_emission0 = np.log(emission0)
        ## M-step
        # pi
        # pi = np.zeros(nstate, dtype=np.float64)
        # log_gamma_list = []
        # for idata in range(ndata):
        #     log_gamma_list.append(log_alpha_list[idata] + log_beta_list[idata])
        #     pi = pi + np.exp(log_gamma_list[idata][0, :])
        # pi = pi/np.sum(pi)
        pi = pi0
        # emission
        # emission = np.zeros((nstate, nobs), dtype=np.float64)
        # for idata in range(ndata):
        #     data = data_list[idata]
        #     for istate in range(nstate):
        #         for iobs in range(nobs):
        #             id = (data == iobs)
        #             if np.any(id):
        #                 emission[istate, iobs] = emission[istate, iobs] + np.sum(np.exp(log_gamma_list[idata][id, istate]))
        # emission[np.isnan(emission)] = 0.0
        # emission = emission / np.sum(emission, axis=1)[:, None]
        emission = emission0
        # T
        with timer(' M-step '):
            T = np.zeros((nstate, nstate), dtype=np.float64)
            for idata in range(ndata):
                data = data_list[idata]
                nframe = data.size
                for iframe in range(1, nframe):
                    log_xi = log_alpha_list[idata][iframe-1, :, None] + log_beta_list[idata][iframe, :]
                    T = T + np.exp(log_xi + log_emission0[:, data[iframe]] + log_T0)/factor_list[idata][iframe]
            T[np.isnan(T)] = 0.0
            T = T / np.sum(T, axis=1)[:, None]
        ## Check convergence
        count_iteration += 1
        logL = np.sum(logL)
        check_convergence = np.abs(logL_old - logL)
        print('%d iteration LogLikelihood = %e  delta = %e  tolerance = %e' % (count_iteration, logL, check_convergence, TOLERANCE))
        logL_old = logL
        pi0 = pi
        emission0 = emission
        T0 = T
    return T, emission, pi


def run(gpuid, n_clusters, num, max_iter, use_custom_kernel, output):
    # T = np.array([[0.1, 0.7, 0.2],
    #               [0.2, 0.1, 0.7],
    #               [0.7, 0.2, 0.1]], dtype=np.float64)
    # emission = np.array([[0.9, 0.1],
    #                      [0.6, 0.4],
    #                      [0.1, 0.9]], dtype=np.float64)
    # pi = np.array([8.0, 1.0, 1.0], dtype=np.float64)
    # nframe = 1000
    # state, data_list = msmgenerate(nframe, T, emission, pi)
    # T0 = np.array([[0.1, 0.1, 0.8],
    #               [0.8, 0.1, 0.1],
    #               [0.3, 0.2, 0.5]], dtype=np.float64)
    # emission0 = np.array([[0.9, 0.1],
    #                      [0.6, 0.4],
    #                      [0.1, 0.9]], dtype=np.float64)
    # pi0 = np.array([0.8, 0.1, 0.1], dtype=np.float64)

    matlab = sio.loadmat('test_baumwelch.mat')
    data = np.asarray(matlab['data'], dtype=np.int).flatten() - 1
    T = np.asarray(matlab['T'], dtype=np.float64)
    emission = np.asarray(matlab['emission'], dtype=np.float64)
    pi = np.asarray(matlab['pi_i'], dtype=np.float64).flatten()
    T0 = np.asarray(matlab['T0'], dtype=np.float64)
    data_list = [data]

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
