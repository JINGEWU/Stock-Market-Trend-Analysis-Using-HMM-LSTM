from hmmlearn import hmm


def GMM_HMM(O, lengths, n_states, v_type, n_iter, verbose=True):

    # model = hmm.GMMHMM(n_components=n_states, covariance_type=v_type, n_mix=4, n_iter=n_iter, verbose=verbose).fit(O, lengths)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type=v_type, n_iter=n_iter, verbose=verbose).fit(O, lengths)

    return model
