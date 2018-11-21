
using DelimitedFiles
using Printf

function msmforward(data_list, T, emission, pi_i)
    ndata = length(data_list)
    nstate = length(T[1, :])
    logL = zeros(Float64, ndata)
    alpha_list = []
    factor_list = []
    for idata = 1:ndata
        data = data_list[idata]
        nframe = length(data)
        alpha  = zeros(Float64, (nframe, nstate))
        factor = zeros(Float64, nframe)
        alpha[1, :] = pi_i.*emission[:, data[1]]
        factor[1] = sum(alpha[1, :])
        alpha[1, :] = alpha[1, :]./factor[1]
        for iframe = 2:nframe
            alpha[iframe, :] = sum(alpha[iframe-1, :] .* T, dims=1)' .* emission[:, data[iframe]]
            factor[iframe] = sum(alpha[iframe, :])
            alpha[iframe, :] = alpha[iframe, :]./factor[iframe]
        end
        logL[idata] = sum(log.(factor))
        push!(alpha_list, alpha)
        push!(factor_list, factor)
    end
    logL, alpha_list, factor_list
end

function msmbackward(data_list, factor_list, T, emission, pi_i)
    ndata = length(data_list)
    nstate = length(T[1, :])
    logL = zeros(Float64, ndata)
    beta_list = []
    for idata = 1:ndata
        data   = data_list[idata]
        factor = factor_list[idata]
        nframe = length(data)
        beta   = zeros(Float64, (nframe, nstate))
        beta[nframe, :] .= 1.0
        for iframe = (nframe-1):-1:1
            beta[iframe, :] = sum((T .* (emission[:, data[iframe+1]] .* beta[iframe+1, :])'), dims=2) ./ factor[iframe+1]
        end
        logL[idata] = sum(log.(factor))
        push!(beta_list, beta)
    end
    logL, beta_list
end

function msmbaumwelch(data_list, T0, emission0, pi_i0)
    ## setup
    TOLERANCE = 10.0^(-4)
    check_convergence = Inf64
    count_iteration = 0
    logL_old = 1.0
    #if not isinstance(data_list, list):
    #    data_list = [data_list]
    ndata = length(data_list)
    nobs = length(emission0[1, :])
    nstate = length(T0[1, :])
    T = similar(T0)
    emission = similar(emission0)
    pi_i = similar(pi_i0)
    while check_convergence > TOLERANCE
        ## E-step
        logL, alpha_list, factor_list = msmforward(data_list, T0, emission0, pi_i0)
        print("1"); println(logL)
        logL2, beta_list = msmbackward(data_list, factor_list, T0, emission0, pi_i0)
        print("2"); println(logL2)
        log_alpha_list = []
        for a in alpha_list
            push!(log_alpha_list, log.(a))
        end
        log_beta_list = []
        for b in beta_list
            push!(log_beta_list, log.(b))
        end
        log_T0 = log.(T0)
        log_emission0 = log.(emission0)
        ## M-step
        # pi
        # pi = np.zeros(nstate, dtype=np.float64)
        # log_gamma_list = []
        # for idata in range(ndata):
        #     log_gamma_list.append(log_alpha_list[idata] + log_beta_list[idata])
        #     pi = pi + np.exp(log_gamma_list[idata][0, :])
        # pi = pi/np.sum(pi)
        pi_i = pi_i0
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
        T = zeros(Float64, (nstate, nstate))
        for idata = 1:ndata
          data = data_list[idata]
          nframe = length(data)
          for iframe = 2:nframe
            #log_xi = bsxfun(@plus, log_alpha{idata}(iframe-1, :)', log_beta{idata}(iframe, :));
            log_xi = log_alpha_list[idata][iframe-1, :] .+ log_beta_list[idata][iframe, :]'
            #T = T .+ exp(bsxfun(@plus, log_xi, log_emission0(:, data(iframe))') + log_T0)./factor{idata}(iframe);
            T = T .+ exp.((log_xi .+ log_emission0[:, data[iframe]]') .+ log_T0) ./ factor_list[idata][iframe]
          end
        end
        #T[np.isnan(T)] = 0.0
        T = T ./ sum(T, dims=2)
        ## Check convergence
        count_iteration += 1
        logL = sum(logL)
        check_convergence = abs(logL_old - logL)
        Printf.@printf("%d iteration LogLikelihood = %e  delta = %e  tolerance = %e\n" , count_iteration, logL, check_convergence, TOLERANCE)
        logL_old = logL
        pi_i0 = pi_i
        emission0 = emission
        T0 = T
    end
    T, emission, pi_i
end

function run()
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
    #matlab = sio.loadmat('test_baumwelch.mat')
    #data = np.asarray(matlab['data'], dtype=np.int).flatten() - 1
    data = readdlm("data.dat", ',', Int32)
    #T = np.asarray(matlab['T'], dtype=np.float64)
    T = readdlm("T.dat", ',', Float64)
    #emission = np.asarray(matlab['emission'], dtype=np.float64)
    emission = readdlm("emission.dat", ',', Float64)
    #pi_i = np.asarray(matlab['pi_i'], dtype=np.float64).flatten()
    pi_i = readdlm("pi_i.dat", ',', Float64)
    pi_i = pi_i[:]
    #T0 = np.asarray(matlab['T0'], dtype=np.float64)
    T0 = readdlm("T0.dat", ',', Float64)
    data_list = [data]

    @time T1, emission1, pi1 = msmbaumwelch(data_list, T0, emission, pi_i);
    println(T)
    println(T0)
    println(T1)
end

run()
