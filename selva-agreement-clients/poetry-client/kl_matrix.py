import numpy as np
def kl_matrix(models_names, models_predictions):
    kl_matrix = []
    preds_str_union = [pred['token_str'] for preds in  models_predictions
                            for pred in preds  ]
    print(preds_str_union)
    ordered_vectors = {
            model_name: [0 for _ in range(len(preds_str_union))] for model_name in models_names
            }
    for token_idx, token_str in enumerate(preds_str_union):
        for model_idx in range(len(models_names)):
            model_name = models_names[model_idx]
            preds = models_predictions[model_idx]
            search_for_token = [token_d for token_d in preds
                                if token_d['token_str'] == token_str]
            if len(search_for_token) > 0:
                ordered_vectors[model_name][token_idx] = search_for_token[0]['score']
    for k1 in ordered_vectors.keys():
        row=[]
        for k2 in ordered_vectors.keys():
            # kl_str = str(KL(ordered_vectors[k1],ordered_vectors[k2]))
            kl_pert = KL_with_pertubation(np.array(ordered_vectors[k1]),
                                         np.array(ordered_vectors[k2])
                                         )
            row.append(kl_pert)
        kl_matrix.append(row)
    return np.array(kl_matrix)



def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))



def KL_with_pertubation(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence
