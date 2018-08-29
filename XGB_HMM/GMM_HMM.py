from hmmlearn import hmm
import pickle


def GMM_HMM(O, lengths, n_states, verbose=False):
    # the first step initial a GMM_HMM model
    # input:
    #     O, array, (n_samples, n_features), observation
    #     lengths, list, lengths of sequence
    #     n_states, number of states
    # output:
    #     S, the best state sequence
    #     A, the transition probability matrix of the HMM model

    # model = hmm.GMMHMM(n_components=n_states, n_mix=4, covariance_type="diag", n_iter=1000, verbose=verbose).fit(O, lengths)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000, verbose=verbose).fit(O, lengths)

    pi = model.startprob_
    A = model.transmat_
    _, S = model.decode(O, algorithm='viterbi')
    gamma = model.predict_proba(O)
    pickle.dump(model, open('C:/Users/Administrator/Desktop/HMM_program/save/GMM_HMM_model.pkl', 'wb'))

    return S, A, gamma
