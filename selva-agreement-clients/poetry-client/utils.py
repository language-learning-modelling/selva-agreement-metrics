import os
import json

def dataclass_to_dict(obj):
    return {k:obj.__getattribute__(k)
            for k in obj.__dataclass_fields__.keys()}

def load_config(config_fp_or_jsonstr):
    if os.path.exists(config_fp_or_jsonstr): 
        with open(config_filepath_or_dictstr) as inpf:
            config = json.load(inpf)
            config = {k.upper(): v for k, v in config.items()}
            return config
    else:
        return { k.upper():v for (k,v) in json.loads(config_fp_or_jsonstr).items() } 

