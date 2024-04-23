from llm_agreement_metrics import dataset, metrics, models
# import plotext as plt
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import nltk
import time
from datetime import datetime



def write_obj(data, outfp, batch=True):
    try:
        with open(outfp) as inpf:
         currentDict = json.loads(inpf.read())
    except:
        currentDict = {}
    with open(outfp,'w') as outf:
        if batch:
            for data_dict in data:
                data_id = f"{data_dict['metadata']['pseudo']}_{data_dict['predictions']['maskedTokenIdx']}"
                currentDict.update({data_id: data_dict})
        else:
            currentDict.update({data['metadata']['pseudo']: data})
        updated_dict_str = json.dumps(currentDict,indent=4)
        outf.write(updated_dict_str)

def check_agreement(concat_preds):
    '''
        given two models predictions calculate agreement

        metrics
            concat_preds: list of list of dicts
    '''
    pass

def predictions_matrix(models_predictions, k):
    '''
        Pick the top k predictions of each model
        and create an array of probabilities for each
        model with size 3k (concatenating each 3 models top k)
    '''
    prediction_queries = [[prediction['token_str'] for prediction in model_predictions[:k] ]for model_predictions in models_predictions 
                            ]
    
    prediction_queries = [e for lst in prediction_queries
                            for e in lst]
    concat_preds = []
    for model_predictions in models_predictions:
        concat_predictions = []
        for p_dict in model_predictions:
            if p_dict['token_str'] \
                    in prediction_queries:
                concat_predictions.append(p_dict)
        missing_queries = set(prediction_queries) - set([p['token_str'] for p in concat_predictions])
        concat_predictions = concat_predictions +\
                [{'token_str':q, 'score':0} for q in missing_queries] 
        concat_predictions = sorted(concat_predictions, key= lambda d:d['token_str'])
        print([d['token_str'] for d in concat_predictions])
        concat_preds.append(concat_predictions)
    return concat_preds 

def llm_masked_sentences_per_model(
        model, tokenizer,
        masked_sentence_tokens,
        token_idx,
        nMasks,
        ): 
    maskTokenStr=tokenizer.mask_token
    llm_masked_sentences = [
                dataset.generate_llm_masked_sentence(
                    masked_sentence_tokens=masked_sentence_tokens,
                    token_idx=token_idx,
                    nMasks=nMasks,
                    maskTokenStr=maskTokenStr
                ) for nMasks in range(1,4) 
            ]
    return llm_masked_sentences

def main(config):
    row_dicts = dataset.read_dataset_pandas(config['input_fp'])

    cleanedTexts = [ dataset.clean_text(d['Texte_etudiant'])
                        for d in row_dicts ]
        
    tokenizedTexts = [ dataset.tokenize_text(cleanedText)
                        for cleanedText in cleanedTexts ] 

    models_fps = [
        # '../models/batch-414-bert-base-uncased-fine-tuned-20240305T132046Z-001/'\
        #               'batch-414-bert-base-uncased-fine-tuned',
        '../models/bert-base-uncased-c4200m-unchaged-vocab-73640000/',
        'distilbert-base-uncased',
        'bert-base-uncased',
        #'xlm-roberta-large'
        ]
    models_tpl = models.load_list_of_models(models_fps)
    maxNumOfMasks = 3
    write_batch = []
    prediction_batch = []
    start=datetime.utcnow()
    start_str = f'{start.hour}:{start.minute}:{start.second}'
    print()
    with open(f"./outputs/error_log_{start_str}","w") as errorf:
        pass
    for text_idx, tokenizedText in enumerate(tokenizedTexts):
        row_metadata = row_dicts[text_idx]
        for token_idx, token in enumerate(tokenizedText):
            masked_sentence_tokens = tokenizedText.copy()
            models_predictions = []
            for model_idx, (model, tokenizer) in enumerate(models_tpl):
                llm_masked_sentences = \
                    llm_masked_sentences_per_model(
                            model, tokenizer,
                            masked_sentence_tokens,
                            token_idx,
                            nMasks=maxNumOfMasks,
                            )
                llm_masked_sentence = llm_masked_sentences[0]
                start=time.time()
                try:
                    llm_masked_sentence_predictions = dataset.fill_masks_pipeline(
                            [llm_masked_sentence],
                            model=model,
                            tokenizer=tokenizer,
                            top_k=config['top_k']) 
                except:
                    with open("./outputs/error_log","a") as errorf:
                        errorf.write(llm_masked_sentence+'\n')
                    continue

                models_predictions.append({
                    'model_name': models_fps[model_idx],
                    'predictions': [
                        {
                                k: v
                        for k, v in d.items()
                        if k != 'sequence'
                        } for d in llm_masked_sentence_predictions[:config['top_k']]
                    ]
                    })
            data = {
                    'metadata': row_metadata,
                    'predictions': {
                        'maskedSentenceStr': ''.join(llm_masked_sentence),
                        'maskedTokenIdx': token_idx, 
                        'maskedTokenStr': token,
                        'models': models_predictions
                    },
                    'linguistic_annotations': {
                       'sentence': {
                        },
                       'tokens': {

                        },
                       'models_predictions':[
                            {
                            'model_name': d['model_name'],
                            'predictions_annotations': [{
                                'idx': idx
                            } for idx in range(len(d['predictions']))],
                            } for d in models_predictions 
                        ]
                    }
            }
            write_batch.append(data)
            if len(write_batch) >= 500:
                write_obj(write_batch, outfp=f'./{config["dataset_name"]}_{start_str}.json', batch=True)
                end=datetime.utcnow()
                print(f'{end.hour}:{end.minute}:{end.second}')
                write_batch=[]

    if len(write_batch) > 0:
        write_obj(write_batch, outfp=f'./{config["dataset_name"]}_{start_str}.json', batch=True)
        write_batch=[]
               
if __name__ == '__main__':
    config = {
        'dataset_name': './outputs/selva-learner-predictions',
        'input_fp' : './selva_dataset/celvasp_full_annotated_with_metadata_2018_2023.csv',
        'expected_metadata': [
            'Date_ajout', 'pseudo', 'Voc_range', 'CECRL',
            'nb_annees_L2', 'L1', 'Domaine_de_specialite',
            'Sejours_duree_semaines', 'Sejours_frequence', 'Lang_exposition',
            'L2', 'Note_dialang_ecrit', 'Lecture_regularite',
            'autre_langue', 'tache_ecrit', 'Texte_etudiant', 'Section_renforcee'
        ],
        'top_k': 3
    }
    main(config)
    '''
    concat_preds = predictions_matrix(models_predictions, k)

    # only print masked sentences that has a significant disagreement
    # check_agreement(concat_preds)
    os.system('clear')
    print(''.join(llm_masked_sentence))
    print(len(llm_masked_sentence_predictions))
    top_k_probs = []
    for preds_lst in concat_preds:
        it = []
        for pred_dict in preds_lst:
            it.append(round(pred_dict['score']*100,2))
        print(it)
        top_k_probs.append(it)
    top_k_str = [rag['token_str'] for rag in concat_preds[0]]
    top_k_labels = [i+1 for i in range(k)]
    #plot(models_fps, top_k_probs, top_k_str)
    input()
    '''
