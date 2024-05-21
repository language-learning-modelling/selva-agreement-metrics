import json
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools


def predictions_matrix(models_predictions, k, target_column="token_str"):
    '''
        Pick the top k predictions of each model
        and create an array of probabilities for each
        model with size of the intersection of unique predcitions [max 3k] (concatenating each 3 models top k)
    '''
    prediction_queries = [[prediction[target_column] for prediction in model_predictions[:k] ]for model_predictions in models_predictions 
                            ]
    
    prediction_queries = [e for lst in prediction_queries
                            for e in lst]
    concat_preds = []
    for model_predictions in models_predictions:
        concat_predictions = []
        for p_dict in model_predictions:
            if p_dict[target_column] \
                    in prediction_queries:
                concat_predictions.append(p_dict)
        missing_queries = set(prediction_queries) - set([p[target_column] for p in concat_predictions])
        concat_predictions = concat_predictions +\
                [{target_column:q, 'score':0} for q in missing_queries] 
        concat_predictions = sorted(concat_predictions, key= lambda d:d[target_column])
        print([d[target_column] for d in concat_predictions])
        concat_preds.append(concat_predictions)
    return concat_preds 

def plot(models_fps, top_k_probs, top_k_str):
    '''
    given a list of models that for each you have
    a list of probabilities plot their probabilities
    grouped by token 

    plt.simple_multiple_bar(top_k_str,
                   top_k_probs,
                   width= 0.1,
                   )
    plt.title(f'model top {k}')
    plt.show() 
    '''
    models_names = [
            m.replace('c4200m','full-efcamdat').split('/')[-1][5:-15] 
                if len(m) > 50
                else 
            m.replace('c4200m','full-efcamdat').split('/')[-1] 
                for m in models_fps
            ]
    model_names = [
            'learner-model-full-efcamdat',
            'native-model-bert'
            ]
    tokens = top_k_str 
    probs_per_model = {
        model_name : probs_lst
            for model_name, probs_lst
                in zip(models_names,
                        top_k_probs) 
    }

    x = np.arange(len(tokens))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in probs_per_model.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Model probabilities by tokens')
    ax.set_xticks(x + width, tokens)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.savefig(f"/home/berstearns/Desktop/image.png")
    plt.show()


def aggregate_dicts(lst_of_lst_of_dicts, target_column):
    '''
     aggregate for each model in a list of models
      the list of predictions by a target column
    '''
    models_aggregates = []
    for model_predictions in lst_of_lst_of_dicts:
        model_aggregate = {}
        for prediction_dict in model_predictions:
            if model_aggregate.get(prediction_dict["ud_pos"]): 
                model_aggregate[prediction_dict["ud_pos"]]["score"] += prediction_dict["score"]
            else:
                model_aggregate[prediction_dict["ud_pos"]] = {
                            "ud_pos": prediction_dict["ud_pos"],
                            "score" : prediction_dict["score"]
                        }
        models_aggregates.append(model_aggregate.values())
    return models_aggregates



def plot_predictions(concat_preds):
    top_k_probs = []
    for preds_lst in concat_preds:
        it = []
        for pred_dict in preds_lst:
            percentage_float = float(pred_dict['score'])*100
            it.append(round(percentage_float,2))
        top_k_probs.append(it)
    top_k_str = [rag['token_str'] for rag in concat_preds[0]]
    top_k_labels = [i+1 for i in range(config["TOP_K"])]
    plot(config["MODELS_FPS"], top_k_probs, top_k_str)


def plot_pos(concat_pos):
    top_k_probs = []
    for preds_lst in concat_pos:
        it = []
        for pred_dict in preds_lst:
            percentage_float = float(pred_dict['score'])*100
            it.append(round(percentage_float,2))
        top_k_probs.append(it)
    top_k_str = [rag['ud_pos'] for rag in concat_pos[0]]
    top_k_labels = [i+1 for i in range(config["TOP_K"])]
    plot(config["MODELS_FPS"], top_k_probs, top_k_str)


def calculate_intersection_matrix(models_predictions):
    '''
     calculate a simmetric matrix n-models x n-models
     calculating the intersection count between two given models
    ''' 
    intersection_matrix = []
    n_models = len(models_predictions)
    for m1_preds in list(models_predictions):
        row=[]
        for m2_preds in list(models_predictions):
            top_k_tokens_m1 = set([d['token_str'] for d in m1_preds])
            top_k_tokens_m2 = set([d['token_str'] for d in m2_preds])
            intersection_count = len(top_k_tokens_m1.intersection(top_k_tokens_m2)) 
            row.append(intersection_count)
        intersection_matrix.append(row)
    return intersection_matrix
            
    
if __name__ == "__main__":
    config = {
            "TOP_K": 3,
            "MODELS_FPS": ["bert-base-uncased","bert-c4_200m","bert-efcamdat"]
    }
    with open("./sample_for_analytics.json") as inpf:
        masked_sentences = json.load(inpf)

    global_stats = {
            "intersection_matrix": np.zeros((3,3))
            }
    for pseudo_id, masked_sentence_dict in masked_sentences.items():
        stats = {
                "intersection_matrix": None,
        }
        d = masked_sentence_dict
        models_predictions = [model_d["predictions"] for model_d in d["predictions"]["models"]]
        concat_preds = predictions_matrix(models_predictions, config["TOP_K"])
        concat_pos = aggregate_dicts(predictions_matrix([model_d["predictions"] for model_d in d["predictions"]["models"]], config["TOP_K"], target_column="ud_pos"), target_column="ud_pos")
        # only print masked sentences that has a significant disagreement
        # check_agreement(concat_preds)
        os.system('clear')
        print(d['predictions']['maskedSentenceStr'])
        plot_predictions(concat_preds)
        input()
        os.system('clear')
        print(d['predictions']['maskedSentenceStr'])
        plot_pos(concat_pos)
        input()

        stats["intersection_matrix"] =  np.array(calculate_intersection_matrix(models_predictions))
        global_stats["intersection_matrix"] = global_stats["intersection_matrix"] + stats["intersection_matrix"]  
        print(stats["intersection_matrix"])
        print(global_stats["intersection_matrix"])
        input()
    print(global_stats["intersection_matrix"])