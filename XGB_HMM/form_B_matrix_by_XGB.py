import xgboost as xgb


def form_B_matrix_by_XGB(model, O, pi):
    # input:
    #     model, the model
    #     O, array, (n_samples, n_features), observation
    #     pi, array, (n_states, ), prior prob of states, P(S)
    # output:
    #     B, array, (n_samples, n_states), P(O|S)

    pred = model.predict(xgb.DMatrix(O))

    B = pred/pi

    return B
