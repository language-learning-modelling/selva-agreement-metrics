'''
    load models to be used to create metrics
'''
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM



def load_list_of_models(model_paths):
    '''
        given a list of remote and/or local models
        load them with their respective tokenizer
    '''
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
    # for path in range(model_paths):
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    return [(model,tokenizer)]
