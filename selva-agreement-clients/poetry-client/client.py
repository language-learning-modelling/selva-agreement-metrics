from llm_agreement_metrics import dataset, metrics, models
# import plotext as plt
import os
import random
import json
import matplotlib.pyplot as plt
import numpy as np



def write_obj(data_dict, outfp):
    with open(outfp) as inpf:
        try:
            currentDict = json.loads(inpf.read())
        except:
            currentDict = {}
    with open(outfp,'w') as outf:
        currentDict.update({random.randint(1,10000000): data_dict})
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

def main():
    row_tpls = dataset.read_dataset_pandas('./selva_dataset/celvasp_full_annotated_with_metadata_2018_2023.csv')

    cleanedTexts = [ dataset.clean_text(rowText)
                        for (rowVR, rowText) in row_tpls ]
        
    tokenizedTexts = [ dataset.tokenize_text(cleanedText)
                        for cleanedText in cleanedTexts ] 

    models_fps = [
        # '../models/batch-414-bert-base-uncased-fine-tuned-20240305T132046Z-001/'\
        #               'batch-414-bert-base-uncased-fine-tuned',
        '../models/bert-base-uncased-c4200m-unchaged-vocab-73640000-20240305T133611Z-001/bert-base-uncased-c4200m-unchaged-vocab-73640000',
        'distilbert-base-uncased',
        #'bert-base-uncased',
        #'xlm-roberta-large'
        ]
    models_tpl = models.load_list_of_models(models_fps)
    maxNumOfMasks = 3
    for text_idx, tokenizedText in enumerate(tokenizedTexts):
        row_metadata = row_tpls[text_idx]
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
                llm_masked_sentence_predictions = dataset.fill_masks_pipeline(
                        [llm_masked_sentence],
                        model=model,
                        tokenizer=tokenizer) 
                models_predictions.append({
                    'model_name': models_fps[model_idx],
                    'predictions': llm_masked_sentence_predictions
                    })
                print(dir(tokenizer))
                print(llm_masked_sentence_predictions)
                print(row_metadata)
                input()
            k=3
            data = {
                    'maskedSentenceStr': ''.join(llm_masked_sentence),
                    'maskedTokenStr': token,
                    'models': models_predictions
            }
            write_obj(data, outfp='./dataset.json')
           
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
        

if __name__ == '__main__':
    main()
