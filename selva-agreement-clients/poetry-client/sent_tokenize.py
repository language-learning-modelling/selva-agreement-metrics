from llm_agreement_metrics import dataset, metrics, models
# import plotext as plt
import sys
import os
import random
import json
import numpy as np
import nltk
import time
import spacy_udpipe
import nltk
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool




def write_obj(data, outfp, batch=True):
    current_stored_dict={}
    with open(outfp,'w') as outf:
        if batch:
            for data_dict in data:
                data_id = data_dict['text_idx']
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

def pos_annotate_prediction(params_dict):
    prediction_dict, masked_sentence_tokens, token_idx = params_dict.values()
    masked_sentence_tokens[token_idx] = prediction_dict["token_str"]
    simulated_text = " ".join(masked_sentence_tokens) 
    prediction_dict["ud_pos"] = dataset.tokenize_text(simulated_text, config['ud_model'])[token_idx].pos_
    return prediction_dict

def main(config):
    main_start=datetime.utcnow()
    print(f'main script: {main_start.year}-{main_start.month}-{main_start.day}_{main_start.hour}:{main_start.minute}:{main_start.second}')
    row_dicts = dataset.read_dataset(config['input_fp'])
    text_column = 'text'#'Texte_etudiant'

    start=datetime.utcnow()
    start_str = f'{start.year}-{start.month}-{start.day}_{start.hour}:{start.minute}:{start.second}'
    write_batch = []
    prediction_batch = []
    print('starting processing',start_str)

    pbar = tqdm(row_dicts)
    pbar.set_description(f"{config['input_filename']}")
    for text_idx, row_dict in enumerate(pbar):
        tokenize_params = {
                "text":  row_dict[text_column],
                "model": config['ud_model']
                }
        tokenizedText = dataset.tokenize_text(tokenize_params)
        tokenizedText = [{'token_str':t.text,'ud_pos':t.pos_ } for t in tokenizedText] 
        text_data = {
                "text_idx": text_idx,
                "text": row_dict[text_column],
                "text_metadata": {
                    k:v for k,v in row_dict.items() if k != text_column
                },
                "sentences": [],
                "tokens": []
        }

        for token_idx, token in enumerate(tokenizedText):
            text_id = row_dict['pseudo'] if row_dict.get('pseudo') else text_idx 
            maskedTokenId = f"{text_id}_{token_idx}"

            models_predictions = []
            data = {
                    'token': token,
                    'predictions': {
                        'maskedTokenId':  maskedTokenId,
                        'maskedTokenIdx': token_idx, 
                        'maskedTokenStr': token['token_str'],
                        'models': models_predictions
                    }
            }
            text_data["tokens"].append(data)
            write_batch.append(text_data)

    if len(write_batch) > 0:
        write_obj(write_batch, outfp=config['output_fp'], batch=True)
        write_batch=[]
               
if __name__ == '__main__':
    config = {
        #'input_fp' : './outputs/EFCAMDAT/train_efcamdat_splits/train_cleaned_efcamdat__all_segment_al',  
        'ud_model_fp': './udpipe_models/english-ewt-ud-2.5-191206.udpipe'
    }

    ud_model = spacy_udpipe.load_from_path(lang="en",
                                      path=config["ud_model_fp"],
                                      meta={"description": "A4LL suggested model"})
    config['input_fp'] = sys.argv[1]
    config['ud_model'] = ud_model
    config['input_filename'] = config['input_fp'].split('/')[-1] 
    config['output_fp'] = f'./outputs/EFCAMDAT/tokenization_batch/train/'\
                          f'{config["input_filename"]}'
    print(config['output_fp'])
    if not config.get('input_fp') and not os.path.exists(config['input_fp']):
        raise Exception("Missing input file")
    main(config)
