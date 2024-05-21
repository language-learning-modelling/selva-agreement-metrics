import re
import csv
import pandas as pd
import nltk
import spacy_udpipe
from transformers import pipeline

def generate_llm_masked_sentence(
        masked_sentence_tokens,
        token_idx,
        nMasks,
        maskTokenStr
        ):
    '''
        generate a masked sentence with
        flexible numbers of masked tokens
        for a target token that needs to be masked
    '''
    repeatedMaskTokens = ''.join([maskTokenStr for _ in range(nMasks)])
    masked_sentence_tokens[token_idx] = repeatedMaskTokens
    masked_sentence_str = ' '.join(masked_sentence_tokens)
    return masked_sentence_str

def fill_masks_pipeline(sentences_lst, model, tokenizer, top_k):
    fill_masks_pipeline = pipeline(task='fill-mask',
                                   model=model,
                                   tokenizer=tokenizer,
                                   top_k=top_k,
                                   device=0,
                                   )
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}
    #tokenizedTexts = [ [ {'token_str':t.text,'ud_pos':t.pos_ } for t in tokenLst]
    #                       for tokenLst in tokenizedTexts] 
    preds = fill_masks_pipeline(
                              sentences_lst,
                              tokenizer_kwargs
                              )
    return preds

def clean_text(rowText):
    '''
        INPUT:
            a raw text string
            rowText : str
        OUTPUT:
            returns alphanumeric 
            []
    '''
    cleanedText = re.sub('[^a-zA-Z.()^,!?+*&%$#"_/<>\']', ' ', rowText)
    cleanedText=rowText
    return cleanedText

def tokenize_text(cleanedText,
                  model):
    '''
        INPUT:
            cleanedText : str
            model: ud_model
        OUTPUT:
            [Tokens]
    '''
    doc = model(cleanedText)
    tokens = [ token for token in doc ]
    return tokens 

def read_dataset_pandas(
                        filepath, 
                        targetL2=['Anglais']
                       ):
    dataset = pd.read_csv(filepath)
    ds_eng = dataset.loc[ dataset['L2'].isin(targetL2) ]
    ds_eng = ds_eng 
    texts = ds_eng['Texte_etudiant'].to_list()
    vocrange = ds_eng['Voc_range'].to_list()
    records = dataset.reset_index().to_dict(orient='records')
    return records

def read_dataset(filepath):
    with open(filepath) as inpf:
        columns_name = next(inpf).replace('\n','').split(',')
        rows=[]
        row_content=''
        for line in inpf:
            row_content += line
            if re.search('",[01]$', line):
                rows.append(row_content)
                row_content=''

        for row_idx, row in enumerate(csv.reader(rows, quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)):
            if len(row) != 17:
                input()
