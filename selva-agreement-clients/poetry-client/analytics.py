import json
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import collections
import pandas as pd
from termcolor import colored
from kl_matrix import kl_matrix
from visualisations import groups_box_plot


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
        #print([d[target_column] for d in concat_predictions])
        concat_preds.append(concat_predictions)
    return concat_preds

def plot(top_k, models_names, top_k_probs, top_k_str, target_column, ax):
    '''
    given a list of models that for each you have
    a list of probabilities plot their probabilities
    grouped by token or other column

    plt.simple_multiple_bar(top_k_str,
                   top_k_probs,
                   width= 0.1,
                   )
    plt.title(f'model top {k}')
    plt.show()
    '''
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

    for attribute, measurement in probs_per_model.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Probabilities in %')
    ax.set_title(f'Model top {top_k} probabilities of predicting a {target_column}')
    ax.set_xticks(x + width, tokens)
    ax.set_ylim(0, 100)
    # plt.savefig(f"")
    #plt.show()


def aggregate_dicts(lst_of_lst_of_dicts, target_column):
    '''
     aggregate for each model in a list of models
      the list of predictions by a target column
    '''
    models_aggregates = []
    for model_predictions in lst_of_lst_of_dicts:
        model_aggregate = {}
        for prediction_dict in model_predictions:
            if model_aggregate.get(prediction_dict[target_column]):
                model_aggregate[prediction_dict[target_column]]["score"] += prediction_dict["score"]
            else:
                model_aggregate[prediction_dict[target_column]] = {
                            target_column: prediction_dict[target_column],
                            "score" : prediction_dict["score"]
                        }
        models_aggregates.append(model_aggregate.values())
    return models_aggregates

def agreement_plot(concat_pos, models_names, top_k, target_column, fig, idx):
    ax = fig.add_subplot(1, 2, idx)  # row 1, column 2, count 1
    top_k_probs = []
    for preds_lst in concat_pos:
        it = []
        for pred_dict in preds_lst:
            percentage_float = float(pred_dict['score'])*100
            it.append(round(percentage_float,2))
        top_k_probs.append(it)
    top_k_str = [rag[target_column] for rag in concat_pos[0]]
    top_k_labels = [i+1 for i in range(top_k)]
    if idx == 1:
        ax = plot(top_k, models_names, top_k_probs, top_k_str, target_column, ax)
    else:
        ax = plot(top_k, models_names, top_k_probs, top_k_str, target_column, ax)
    return fig

def plot_all(
        fig_width,
        fig_height,
        models_names,
        top_k,
        targets
        ):

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.2, wspace=1)
    for (idx, tpl) in enumerate(targets):
        fig = agreement_plot(
                    tpl[1],
                    models_names=models_names,
                    top_k=top_k,
                    target_column=tpl[0],
                    fig=fig,
                    idx=idx+1)
    plt.tight_layout()
    plt.legend(loc='upper left', ncols=2)
    plt.show()

def calculate_disagreement_score(m1_preds, m2_preds, intersection_tokens):
    if len(intersection_tokens) == 0:
        disagreement_score = 1

    for intersection_token in intersection_tokens:
        pred_dicts_m1 = [d for d in m1_preds if d['token_str'] == intersection_token ]
        pred_score_m1 = pred_dicts_m1[0]['score']
        pred_dicts_m2 = [d for d in m2_preds if d['token_str'] == intersection_token ]
        pred_score_m2 = pred_dicts_m2[0]['score']
        disagreement_score = abs(pred_score_m1-pred_score_m2)
    
    return disagreement_score

def calculate_metrics_between_models(models_predictions):
    '''
     calculate one simmetric matrix n-models x n-models
     for each intersection metric to be calculated between two given models
    '''
    intersection_count_matrix = []
    intersection_disagreement_matrix = []
    n_models = len(models_predictions)
    for m1_preds in list(models_predictions):
        count_row=[]
        disagreement_row=[]
        for m2_preds in list(models_predictions):
            top_k_tokens_m1 = set([d['token_str'] for d in m1_preds])
            top_k_tokens_m2 = set([d['token_str'] for d in m2_preds])
            intersection_tokens = top_k_tokens_m1.intersection(top_k_tokens_m2)
            intersection_count = len(intersection_tokens)
            intersection_disagreement_score = calculate_disagreement_score(
                                                m1_preds, m2_preds,
                                                intersection_tokens)
            count_row.append(intersection_count)
            disagreement_row.append(intersection_disagreement_score)
        intersection_count_matrix.append(count_row)
        intersection_disagreement_matrix.append(disagreement_row)
    return np.array(intersection_count_matrix), np.array(intersection_disagreement_matrix)

def calculate_learner_behavior_metrics(models_predictions, learner_actual_token_str):
    '''
     calculate one simmetric matrix n-models x n-models
     for each intersection metric to be calculated between two given models
    '''
    scores_per_model = []
    n_models = len(models_predictions)
    for m_preds in list(models_predictions):
        search_for_token = [ d for d in m_preds if d['token_str'].lower() == learner_actual_token_str.lower()]
        actual_token_score = search_for_token[0]['score'] if len(search_for_token) > 0 else 0 
        scores_per_model.append(actual_token_score)
    return np.array(scores_per_model)

if __name__ == "__main__":
    config = {
            "INPUT_FP": "./outputs/selva-learner-predictions_2024-6-6_14:24:5.json",#"./outputs/CELVA/celva-predictions.json",
            "LEXICAL_FP": "./outputs/SUBTLEXusfrequencyabove1.xls",
            "TOP_K": 10,
            "MODELS_NAMES": ["bert-base-uncased","bert-c4_200m","bert-efcamdat"],
            "FIG_WIDTH": 10,
            "FIG_HEIGHT": 8
    }

    bug_errors_count = {
            'missing_predictions': 0
    }
    lexical_ref = json.loads(
                    pd.read_excel(config["LEXICAL_FP"]).to_json(orient="records")
                  )  
    with open(config["INPUT_FP"]) as inpf:
        masked_sentences = json.load(inpf)

    global_stats = {
            "intersection_matrix": np.zeros((3,3))
            }
    texts_aggregations = collections.defaultdict(lambda : {
            "n_of_tokens": 0,
        })
    text_stats = {
            "n_of_tokens": 0,
            "intersection_matrix": None,
    }
    previous_text_pseudo_id = None
    cefr_kl_boxplot_data= {
            'A1': [],
            'A2': [],
            'B1': [],
            'B2': [],
            'C1': [],
            'C2': [],
    }
    voc_range_kl_boxplot_data= {
            'A1': [],
            'A2': [],
            'B1': [],
            'B2': [],
            'C1': [],
            'C2': [],
    }
    cefr_kl_ud_pos_boxplot_data=collections.defaultdict(lambda : {
                'A1': [],
                'A2': [],
                'B1': [],
                'B2': [],
                'C1': [],
                'C2': [],
    })
    voc_range_kl_ud_pos_boxplot_data=collections.defaultdict(lambda : {
                'A1': [],
                'A2': [],
                'B1': [],
                'B2': [],
                'C1': [],
                'C2': [],
    })
    cefr_prob_diff_boxplot_data= {
            'A1': [],
            'A2': [],
            'B1': [],
            'B2': [],
            'C1': [],
            'C2': [],
    }
    cefr_prob_diff_ud_pos_boxplot_data=collections.defaultdict(lambda : {
                'A1': [],
                'A2': [],
                'B1': [],
                'B2': [],
                'C1': [],
                'C2': [],
    })
    voc_range_prob_diff_boxplot_data= {
            'A1': [],
            'A2': [],
            'B1': [],
            'B2': [],
            'C1': [],
            'C2': [],
    }
    for masked_sentence_pseudo_id, masked_sentence_dict in masked_sentences.items():
        text_pseudo_id = masked_sentence_pseudo_id.split("_")[0]
        if not (previous_text_pseudo_id is None) and\
                previous_text_pseudo_id != text_pseudo_id:
            texts_aggregations[previous_text_pseudo_id].update(text_stats)
            # print(texts_aggregations[previous_text_pseudo_id]);input()
            text_stats = {
                    "n_of_tokens": 0,
                    "intersection_matrix": None,
            }

        d = masked_sentence_dict
        text_cefr = d['metadata']['CECRL']
        text_vocab_range = d['metadata']['Voc_range']
        masked_token_str = d['predictions']['maskedToken']['token_str']
        masked_token_ud_pos = d['predictions']['maskedToken']['ud_pos']
        models_names = [model_d["model_name"] for model_d in d["predictions"]["models"]]
        models_predictions = [model_d["predictions"] for model_d in d["predictions"]["models"]]
        concat_preds = predictions_matrix(models_predictions, config["TOP_K"])
        concat_pos = aggregate_dicts(
                predictions_matrix([model_d["predictions"] for model_d in d["predictions"]["models"]], config["TOP_K"], target_column="ud_pos"),
                target_column="ud_pos"
                )

        ############################################
        ##                                        ##
        ##    leanrner used token metrics         ##
        ##                                        ##
        ############################################
        '''
        # what is the model more likely to use the learner token ?
        '''
        learner_metrics =  calculate_learner_behavior_metrics(
                models_predictions,
                masked_token_str
                )
        try:
            native_full_learner_prob_diff = abs(learner_metrics[0])
            cefr_prob_diff_boxplot_data[text_cefr].append(native_full_learner_prob_diff)
            cefr_prob_diff_ud_pos_boxplot_data[masked_token_ud_pos][text_cefr].append(native_full_learner_prob_diff)
            voc_range_prob_diff_boxplot_data[text_cefr].append(native_full_learner_prob_diff)
        except:
            pass

        ############################################
        ##                                        ##
        ##    model agreement metrics             ##
        ##                                        ##
        ############################################
        intersection_matrix_count,\
                intersection_matrix_score =  calculate_metrics_between_models(models_predictions)
        print(intersection_matrix_count)
        print(intersection_matrix_score)

        kl_metric_matrix = kl_matrix(models_names, models_predictions)
        try:
            arbitrary_kl_metric = kl_metric_matrix[0][2] + kl_metric_matrix[2][0]
            cefr_kl_boxplot_data[text_cefr].append(arbitrary_kl_metric)
            cefr_kl_ud_pos_boxplot_data[masked_token_ud_pos][text_cefr].append(arbitrary_kl_metric)
            voc_range_kl_boxplot_data[text_vocab_range].append(arbitrary_kl_metric)
            voc_range_kl_ud_pos_boxplot_data[masked_token_ud_pos][text_vocab_range].append(arbitrary_kl_metric)
        except:
           bug_errors_count['missing_predictions'] += 1

        ############################################
        ##                                        ##
        ##    text level metrics                  ##
        ##                                        ##
        ############################################
        text_stats["n_of_tokens"] += 1
        #text_stats["intersection_matrix_count"], text_stats["intersection_matrix_score"] =\
        #                    intersection_matrix_count, intersection_matrix_score
        # text_stats["intersection_matrix_kl"] =  np.array(calculate_intersection_matrix(models_predictions))

        # global_stats["intersection_matrix"] = global_stats["intersection_matrix"] + text_stats["intersection_matrix"]

        print("masked sentence string")
        print("*"*30)

        colored_masked_sentence_str = d['predictions']['maskedSentenceStr']
        colors_splits = colored_masked_sentence_str.split("[MASK]")
        p1 = colors_splits[0]
        if len(colors_splits) > 1:
            p2 = colors_splits[1]
        else:
            p2 = ''
        print(colored(p1,'white'), colored('[MASK]','green'), colored(p2,'white'))

        '''
        plot_all(
                fig_width=config["FIG_WIDTH"],
                fig_height=config["FIG_HEIGHT"],
                models_names=config["MODELS_NAMES"],
                top_k=config["TOP_K"],
                targets=[
                    ("token_str", concat_preds),
                    ("ud_pos", concat_pos),
                    ]
                )
        '''
        #input()
        previous_text_pseudo_id = text_pseudo_id
    # print(global_stats["intersection_matrix"])
    # print(f'bug errors count, missing predictions: {bug_errors_count["missing_predictions"]}') 
