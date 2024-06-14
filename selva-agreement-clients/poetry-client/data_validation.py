import json
import pydantic


class CelvaProcessedInstanceV1(pydantic.BaseModel):
    metadata: dict
    predictions: dict
    linguistic_annotations: None



def check_predictions_have_all_expected_models(celva_instance_dict, expected_models_names):
    pass

def check_predictions_have_expected_top_k_predictions(celva_instance_dict):
    for model_dict in celva_instance_dict['predictions']['models']:
        print(len(model_dict['predictions']))
        input()



if __name__ == "__main__":
    config = {
            "INPUT_FP": "./outputs/selva-learner-predictions_2024-6-13_16:18:18.json"
    }
    with open(config["INPUT_FP"]) as inpf:
        processed_selva_dict = json.load(inpf)
    for maskedTokenId, instance_dict in processed_selva_dict.items():
        check_predictions_have_expected_top_k_predictions(instance_dict)
