import json
import collections
import os
import sys
import pathlib
import srsly
import re

def main():
    MAXK=100
    top_k_count = collections.defaultdict(lambda: collections.defaultdict(int))
    #"bert-base-uncased"
    #"bert-base-uncased-c4200m-unchaged-vocab-73640000"
    #"bert-base-uncased"
    #"roberta-base""
    #"mosaic-bert-base"
    TARGET_MODEL_NAME="bert-base-uncased"
    DATASET="CELVA"
    SPLIT=""
    DATA_FOLDERPATH=f"/home/berstearns/projects/language-learning-modelling/selva-agreement-metrics/selva-agreement-clients/poetry-client/outputs/"
    TARGET_MODEL_FOLDERPATH=f"{DATA_FOLDERPATH}/{DATASET}/finalized/{TARGET_MODEL_NAME}/"
    total_n_tokens = 0
    CEFR_COLUMN="CECRL" 
    #"CECRL"
    #"cefr" 
    '''
    segment_search = re.compile(r"segment_([a-z]+).json_"+TARGET_MODEL_NAME)
    expected_files = collections.defaultdict(list)
    for p in pathlib.Path(TARGET_MODEL_FOLDERPATH).iterdir():
        search = segment_search.search(p.as_posix())
        if search:
            segment = search.group(1)
            expected_files[segment].append(p.as_posix())
        else:
            raise Exception(f"Unknown file format {p.as_posix()}")
    for expected_segment, expected_files_lst in expected_files.items():
        print(len(expected_files_lst))
    '''

    for filepath in pathlib.Path(TARGET_MODEL_FOLDERPATH).iterdir():
        inputfp = filepath.as_posix()
        print(inputfp)
        with open(inputfp) as inpf:
            if inputfp.endswith("json.gz") or inputfp.endswith(".json.compact.gz"):
                text_dicts_dict = srsly.read_gzip_json(inputfp)
            elif inputfp.endswith(".json.compact"):
                text_dicts_dict = json.load(inpf)
            else:
                raise Exception(f"Unknown file format {filepath}")
            for text_id in text_dicts_dict: 
                cefr_level = text_dicts_dict[text_id]["text_metadata"][CEFR_COLUMN]
                for token_dict in text_dicts_dict[text_id]["tokens"]:
                    token_has_prediction = token_dict["predictions"]["models"].get(TARGET_MODEL_NAME,False)
                    if not token_has_prediction:
                        break
                    total_n_tokens+=1
                    maskedTokenStr = token_dict["predictions"]["maskedTokenStr"] 
                    tokenUdPos = token_dict["token"]["ud_pos"]
                    topMaxK_dicts_lst = token_dict["predictions"]["models"][TARGET_MODEL_NAME]
                    search = [{"rank":idx+1,
                            "data": d} for idx, d in enumerate(topMaxK_dicts_lst) if maskedTokenStr.lower() == d['token_str'].lower()]   
                    if search:
                        prediction = search[0]
                        for k in [1,5,10,25,50,100]:
                            if prediction["rank"] <= k:
                                top_k_count["total"][k] += 1
                                top_k_count[tokenUdPos][k] += 1
                                top_k_count[cefr_level][k] += 1
                    top_k_count["total"]["count"]   +=1
                    top_k_count[tokenUdPos]["count"]+=1
                    top_k_count[cefr_level]["count"]+=1
            del text_dicts_dict
            top_k_count_fp = f"./results/top_k_count_{DATASET}_{SPLIT}_{TARGET_MODEL_NAME}.json"\
                                    if SPLIT else f"./results/top_k_count_{DATASET}_{TARGET_MODEL_NAME}.json"
            with open(top_k_count_fp,"w") as outf:
                json.dump(top_k_count,outf)
        top_k_proportions={}
        for dict_id, top_k_count_dict in top_k_count.items(): 
            total_n_tokens_in_category=top_k_count_dict["count"]
            top_k_proportions[dict_id] =\
                {k:v/total_n_tokens_in_category for k,v in top_k_count_dict.items()}
        top_k_proportions_fp = f"./results/top_k_proportions_{DATASET}_{SPLIT}_{TARGET_MODEL_NAME}.json"\
                                if SPLIT else f"./results/top_k_proportions_{DATASET}_{TARGET_MODEL_NAME}.json"
        with open(top_k_proportions_fp,"w") as outf:
            json.dump(top_k_proportions,outf)

if __name__ == "__main__":
    main()
