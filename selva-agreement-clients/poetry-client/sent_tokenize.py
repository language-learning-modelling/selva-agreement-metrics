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



from dataclasses import dataclass

@dataclass
class Config:
    INPUT_FP: str = None
    OUTPUT_FOLDER: str = None
    UD_MODEL_FP: str = None
    TEXT_COLUMN: str = None

    def __post_init__(self):
        for field_key in self.__dataclass_fields__.keys():
            if self.__getattribute__(field_key) is None:
             raise ValueError(f'missing {field_key} config property')


def write_obj(data, outfp, batch=True):
    current_stored_dict={}
    with open(outfp,'w') as outf:
        if batch:
            for data_dict in data:
                data_id = data_dict['text_id']
                current_stored_dict.update({data_id: data_dict})
        else:
            data_id = data['metadata']['pseudo']
            current_stored_dict.update({data['metadata']['pseudo']: data})
        updated_dict_str = json.dumps(current_stored_dict,indent=4)
        outf.write(updated_dict_str)

def main(config):
    main_start=datetime.utcnow()
    print(f'main script: {main_start.year}-{main_start.month}-{main_start.day}_{main_start.hour}:{main_start.minute}:{main_start.second}')
    row_dicts = dataset.read_dataset(config.INPUT_FP)

    start=datetime.utcnow()
    start_str = f'{start.year}-{start.month}-{start.day}_{start.hour}:{start.minute}:{start.second}'
    write_batch = []
    prediction_batch = []
    print('starting processing',start_str)


    '''
    pbar = tqdm(row_dicts)
    pbar.set_description(f"{config.INPUT_FILENAME}")
    for text_idx, row_dict in enumerate(pbar):
        tokenize_params = {
                "text":  row_dict[text_column],
                "model": config.UD_MODEL
                }
    '''
    pbar = tqdm(row_dicts.items())
    pbar.set_description(f"{config.INPUT_FILENAME}")
    for text_idx, (text_id, row_dict) in enumerate(pbar):
        tokenize_params = {
                "text":  row_dict[config.TEXT_COLUMN],
                "model": config.UD_MODEL
                }
        tokenizedText = dataset.tokenize_text(tokenize_params)
        tokenizedText = [{'token_str':t.text,'ud_pos':t.pos_ } for t in tokenizedText] 
        text_data = {
                "text_id": text_id,
                "text": row_dict[config.TEXT_COLUMN],
                "text_metadata": {
                        k:v for k,v in row_dict["text_metadata"].items() if k != config.TEXT_COLUMN
                },
                "sentences": [],
                "tokens": []
        }

        for token_idx, token in enumerate(tokenizedText):
            maskedTokenId = f"{text_id}_{token_idx}"

            models_predictions = {}
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
        write_obj(write_batch, outfp=config.OUTPUT_FP, batch=True)
        write_batch=[]
               
if __name__ == '__main__':
    from utils import load_config, dataclass_to_dict
    
    config_fp_or_jsonstr = "".join(sys.argv[1:])
    config_dict = load_config(config_fp_or_jsonstr) 
    config = Config(**config_dict) 

    ud_model = spacy_udpipe.load_from_path(lang="en",
                                      path=config.UD_MODEL_FP,
                                      meta={"description": "A4LL suggested model"})
    config.UD_MODEL = ud_model
    config.INPUT_FILENAME = config.INPUT_FP.split('/')[-1] 
    config.OUTPUT_FP = f'{config.OUTPUT_FOLDER}/{config.INPUT_FILENAME}'
    main(config)
