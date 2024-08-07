import json
import collections
import os
import srsly

# ./outputs/samples/tnc_c4200m_celva_30_sample.json
#"celva_texts.json_maskedsentences.json_bert-base-uncased.json"
# "celva_texts.json_maskedsentences.json_bert-base-full-efcamdat.json"
#"celva_texts.json_maskedsentences.json_bert-base-uncased_local.json"
# celva_texts.json_maskedsentences.json_bert-base-uncased_local.json
#"celva_texts.json_maskedsentences.json_bert-base-uncased-c4200m-unchaged-vocab-73640000.json"
# "celva_texts.json_maskedsentences.json_bert-base-uncased.json"
# "celva_texts.json_maskedsentences.json_bert-base-uncased-finetuned-cleaned_efcamdat__all.txt_checkpoint-464970.json"
# "bert-base-full-efcamdat"
# "bert-base-uncased"
# "bert-base-uncased-c4200m-unchaged-vocab-73640000"
# "bert-base-uncased-finetuned-cleaned_efcamdat__all.txt_checkpoint-464970"
#"bert-base-uncased"
MAXK=100
top_k_count = collections.defaultdict(lambda: collections.defaultdict(int))
confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int)) 
TARGET_MODEL_NAME="roberta-base"
#"bert-base-uncased"
#"bert-base-uncased-c4200m-unchaged-vocab-73640000"
#"bert-base-uncased"
DATASET="EFCAMDAT"
SPLIT="test"
DATA_FOLDERPATH=f"/home/berstearns/projects/language-learning-modelling/selva-agreement-metrics/selva-agreement-clients/poetry-client/outputs/"
total_n_tokens = 0
CEFR_COLUMN="cefr" 
#"CECRL"
#"cefr" 
expected_roberta_efcamdat_test_files = [
    "test_cleaned_efcamdat__all_segment_aa.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ab.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ac.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ad.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ae.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_af.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ag.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ah.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ai.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_aj.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ak.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_al.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_am.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_an.json_roberta-base.json.gz",
    "test_cleaned_efcamdat__all_segment_ao.json_roberta-base.json.gz"
]
expected_bert_efcamdat_test_files = [
    "test_cleaned_efcamdat__all_segment_aa.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_ab.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_ac.json_bert-base-uncased.json.gz",
    #"test_cleaned_efcamdat__all_segment_ad.json_maskedsentences.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_ae.json_bert-base-uncased.json.gz",
    #"test_cleaned_efcamdat__all_segment_af.json_maskedsentences.json_bert-base-uncased.json.compact",
    #"test_cleaned_efcamdat__all_segment_ag.json_maskedsentences.json_bert-base-uncased.json.compact",
    #"test_cleaned_efcamdat__all_segment_ah.json_maskedsentences.json_bert-base-uncased.json.compact",
    "test_cleaned_efcamdat__all_segment_ai.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_aj.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_ak.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_al.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_am.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_an.json_bert-base-uncased.json.gz",
    "test_cleaned_efcamdat__all_segment_ao.json_bert-base-uncased.json.gz"
]
expected_bert_celva_files = [
        "_celva_texts.json_maskedsentences.json_bert-base-uncased.json"
        ]
dataset_model_files = expected_roberta_efcamdat_test_files
for filename in dataset_model_files:
    print(filename)
    relative_fp=f"{DATASET}/finalized/{TARGET_MODEL_NAME}/{filename}"
    inputfp=os.path.join(DATA_FOLDERPATH,relative_fp)
    with open(inputfp) as inpf:
        try:
            if filename.endswith("json.gz"):
                text_dicts_dict = srsly.read_gzip_json(inputfp)
            elif filename.endswith(".json.compact"):
                text_dicts_dict = json.load(inpf)
            else:
                raise Exception(f"Unknown file format {filename}")
        except:
            print(f"-"*50, "error")
            print(filename)
            print("-"*50)
            continue
        for text_id, text_dict in text_dicts_dict.items(): 
            cefr_level = text_dict["text_metadata"][CEFR_COLUMN]
            for token_dict in text_dict["tokens"]:
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
        top_k_proportions={}
    for dict_id, top_k_count_dict in top_k_count.items(): 
        total_n_tokens_in_category=top_k_count_dict["count"]
        top_k_proportions[dict_id] =\
            {k:v/total_n_tokens_in_category for k,v in top_k_count_dict.items()}
