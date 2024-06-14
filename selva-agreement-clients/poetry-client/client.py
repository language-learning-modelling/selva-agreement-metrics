from llm_agreement_metrics import dataset, metrics, models
# import plotext as plt
import os
import random
import json
import numpy as np
import nltk
import time
from datetime import datetime
import spacy_udpipe



def write_obj(data, outfp, batch=True, partial_dict=None):
    try:
        with open(outfp) as inpf:
         current_stored_dict = json.loads(inpf.read())
    except:
        current_stored_dict = {}
    if not (partial_dict is None):
        missing_items = {
                k:v for k,v in partial_dict.items()
                if k not in current_stored_dict
                }
        current_stored_dict.update(missing_items)
    with open(outfp,'w') as outf:
        if batch:
            for data_dict in data:
                data_id = data_dict['predictions']['maskedTokenId']
                current_stored_dict.update({data_id: data_dict})
        else:
            data_id = data['metadata']['pseudo']
            current_stored_dict.update({data['metadata']['pseudo']: data})
        updated_dict_str = json.dumps(current_stored_dict,indent=4)
        outf.write(updated_dict_str)

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
                ) for nMasks in range(1, nMasks+1) 
            ]
    return llm_masked_sentences

def cls_to_dict(obj):
    key_value_tpls = [(a,getattr(obj, a)) for a in dir(obj) if not a.startswith('__')]
    return dict(key_value_tpls)

def pos_annotate_prediction(prediction_dict, masked_sentence_tokens, token_idx):
    masked_sentence_tokens[token_idx] = prediction_dict["token_str"]
    simulated_text = " ".join(masked_sentence_tokens) 
    prediction_dict["ud_pos"] = dataset.tokenize_text(simulated_text, config['ud_model'])[token_idx].pos_
    return prediction_dict

def main(config):
    main_start=datetime.utcnow()
    print(f'main script: {main_start.year}-{main_start.month}-{main_start.day}_{main_start.hour}:{main_start.minute}:{main_start.second}')
    row_dicts = dataset.read_dataset_pandas(config['input_fp'])


    partial_processed_dict = json.load(open(config['partial_fp'])) if os.path.exists(config['partial_fp']) == True else {}
    cleanedTexts = [ dataset.clean_text(d['Texte_etudiant'])
                        for d in row_dicts ]
        
    tokenizedTexts = [ dataset.tokenize_text(cleanedText, config['ud_model'])
                        for cleanedText in cleanedTexts ] 

    tokenizedTexts = [ [ {'token_str':t.text,'ud_pos':t.pos_ } for t in tokenLst]
                        for tokenLst in tokenizedTexts] 

    models_tpl = models.load_list_of_models(config['models_fps'])
    maxNumOfMasks = 3
    write_batch = []
    prediction_batch = []
    start=datetime.utcnow()
    start_str = f'{start.year}-{start.month}-{start.day}_{start.hour}:{start.minute}:{start.second}'
    write_obj(write_batch, 
            outfp=f'./{config["dataset_name"]}_{start_str}.json',
            batch=True,
            partial_dict=partial_processed_dict
          )
    print('starting processing',start_str)

    with open(f"./outputs/error_log_{start_str}","w") as errorf:
        pass

    for text_idx, tokenizedText in enumerate(tokenizedTexts):
        row_metadata = row_dicts[text_idx]
        for token_idx, token in enumerate(tokenizedText):
            maskedTokenId = f"{row_metadata['pseudo']}_{token_idx}"
            if partial_processed_dict.get(maskedTokenId, False):
                continue

            models_predictions = []
            for model_idx, (model, tokenizer) in enumerate(models_tpl):
                masked_sentence_tokens = [t['token_str'] for t in tokenizedText.copy()]
                llm_masked_sentences = \
                    llm_masked_sentences_per_model(
                            model, tokenizer,
                            masked_sentence_tokens,
                            token_idx,
                            nMasks=1,#maxNumOfMasks,
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
                    'model_name': config['models_fps'][model_idx],
                    'predictions': [
                        pos_annotate_prediction({
                                k: v
                        for k, v in d.items()
                        if k not in ['sequence']
                        }, 
                        masked_sentence_tokens,
                        token_idx) for d in llm_masked_sentence_predictions[:config['top_k']]
                    ]
                    })
            data = {
                    'metadata': row_metadata,
                    'predictions': {
                        'maskedToken': token,
                        'maskedSentenceStr': ''.join(llm_masked_sentence),
                        'maskedTokenId':  maskedTokenId,
                        'maskedTokenIdx': token_idx, 
                        'maskedTokenStr': token['token_str'],
                        'models': models_predictions
                    },
                    'linguistic_annotations': {
                       'sentence': {
                        },
                       'tokens': {
                        },
                       'models_predictions':['''
                            {
                            'model_name': d['model_name'],
                            'predictions_annotations': [{
                                'idx': idx
                            } for idx in range(len(d['predictions']))],
                            } for d in models_predictions''' 
                        ]
                    }
            }
            write_batch.append(data)
            if len(write_batch) >= 10000:
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
        'input_fp' : './outputs/CELVA/celvasp_english_annotated_with_metadata_2018_2023_both_splits_feb2024.csv',
        'partial_fp': './outputs/selva-learner-predictions_2024-6-6_14:24:5.json',
        'expected_metadata': [
            'Date_ajout', 'pseudo', 'Voc_range', 'CECRL',
            'nb_annees_L2', 'L1', 'Domaine_de_specialite',
            'Sejours_duree_semaines', 'Sejours_frequence', 'Lang_exposition',
            'L2', 'Note_dialang_ecrit', 'Lecture_regularite',
            'autre_langue', 'tache_ecrit', 'Texte_etudiant', 'Section_renforcee'
        ],
        'models_fps' : [
            # '../models/batch-414-bert-base-uncased-fine-tuned-20240305T132046Z-001/'\
            #               'batch-414-bert-base-uncased-fine-tuned',
            'bert-base-uncased',
            '../models/bert-base-uncased-c4200m-unchaged-vocab-73640000/',
            '../models/bert-base-uncased-fullefcamdat/',
            #'distilbert-base-uncased',
            #'xlm-roberta-large'
            ],
        'top_k': 100, 
        'ud_model_fp': './udpipe_models/english-ewt-ud-2.5-191206.udpipe'
    }

    ud_model = spacy_udpipe.load_from_path(lang="en",
                                      path=config["ud_model_fp"],
                                      meta={"description": "A4LL suggested model"})
    config['ud_model'] = ud_model
    main(config)
