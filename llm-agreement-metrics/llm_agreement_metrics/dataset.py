import re
import csv
import pandas as pd
import nltk
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

def fill_masks_pipeline(sentences_lst, model, tokenizer):
    fill_masks_pipeline = pipeline(task='fill-mask',
                                   model=model,
                                   tokenizer=tokenizer,
                                   top_k=1000
                                   )
    return fill_masks_pipeline(sentences_lst)

def clean_text(rowText):
    cleanedText = re.sub('[^a-zA-Z.()]', ' ', rowText)
    return cleanedText

def tokenize_text(cleanedText):
    '''
        INPUT:
            cleanedText : str
        OUTPUT:
            [Tokens]
    '''
    return nltk.tokenize.word_tokenize(cleanedText)

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
