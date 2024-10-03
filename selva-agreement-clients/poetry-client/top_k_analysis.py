"""
script to calculate predictions metrics
for MLM.
Calculates top_k_count, proportions and confusion confusion_matrix
"""
from dataclasses import dataclass
import json
import collections
import pathlib
import hashlib
import os
import srsly
from termcolor import colored


@dataclass
class UserConfig:
    """
    Expected user config
    """

    target_model_name: str = None
    dataset: str = None
    data_folderpath: str = None
    cefr_column: str = None
    max_k: int = None
    split: str = ''
    config_hash: str = None
    TARGET_MODEL_NAME: str = None
    DATASET: str = None
    DATA_FOLDERPATH: str = None
    CEFR_COLUMN: str = None
    MAX_K: int = None
    SPLIT: str = ''

    def __post_init__(self):
        # Validate that all required fields are provided
        internal_fields = [
            'config_hash',
            'TARGET_MODEL_NAME',
            'DATASET',
            'DATA_FOLDERPATH',
            'CEFR_COLUMN',
            'MAX_K',
            'SPLIT',
        ]
        for field in self.__dataclass_fields__.keys():
            if (
                field not in internal_fields
                and self.__getattribute__(field) is None
            ):
                raise ValueError(f'missing {field} config property')

        for field in self.__dataclass_fields__.keys():
            self.__setattr__(field.upper(), self.__getattribute__(field))

        self.TARGET_MODEL_FOLDERPATH = (
            f'{self.DATA_FOLDERPATH}/'
            f'{self.DATASET}/finalized/{self.TARGET_MODEL_NAME}/'
        )

        if not pathlib.Path(self.DATA_FOLDERPATH).exists():
            raise ValueError(f'{self.DATA_FOLDERPATH} does not exist')

        if not pathlib.Path(self.TARGET_MODEL_FOLDERPATH).exists():
            raise ValueError(f'{self.TARGET_MODEL_FOLDERPATH} does not exist')

        if self.MAX_K <= 0:
            raise ValueError('Max K must be a positive integer.')

        self.config_hash = self.generate_hash()

    def generate_hash(self):
        """Generate a unique hash for the configuration."""
        config_str = (
            f'{self.target_model_name}_{self.dataset}'
            f'_{self.split}_{self.data_folderpath}_{self.cefr_column}_{self.max_k}'
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_run(self, folderpath):
        """Save the configuration as a JSON file with a unique hash in its name."""
        hash_val = self.config_hash
        run_configs_dir = folderpath
        if not os.path.exists(run_configs_dir):
            os.makedirs(run_configs_dir)

        config_filename = (
            f'config_{self.dataset}_{self.target_model_name}_{hash_val}.json'
            if not self.split == ''
            else f'config_{self.dataset}_{self.split}_{self.target_model_name}_{hash_val}.json'
        )
        config_fp = os.path.join(run_configs_dir, config_filename)
        with open(config_fp, 'w') as config_file:
            json.dump(self.__dict__, config_file, indent=4)
        print(f'Configuration saved to: {config_fp}')


def load_text_dicts(inputfp):
    """Load text dictionaries from a file."""
    if inputfp.endswith('json.gz') or inputfp.endswith('.json.compact.gz'):
        data = srsly.read_gzip_json(inputfp)
    elif inputfp.endswith('.json.compact'):
        with open(inputfp) as inpf:
            data = json.load(inpf)
    else:
        raise ValueError(f'Unknown file format: {inputfp}')
    return data


def process_token_predictions(
    tokens_lst,
    token_dict,
    config,
    top_k_count,
    confusion_matrix,
    cefr_level,
    total_n_tokens,
    n_skipped_tokens,
):
    """Process token predictions and update top_k_count and confusion_matrix."""
    token_has_prediction = token_dict['predictions']['models'].get(
        config.TARGET_MODEL_NAME, False
    )
    if not token_has_prediction:
        n_skipped_tokens += 1
        return total_n_tokens, n_skipped_tokens

    tokens_lst_copy = tokens_lst.copy()
    total_n_tokens += 1
    token_idx = token_dict['predictions']['maskedTokenIdx']
    tokens_lst_copy[token_idx] = colored('[MASK]', 'green')
    masked_sentence_str = ' '.join(tokens_lst_copy)
    masked_token_str = token_dict['predictions']['maskedTokenStr']
    token_ud_pos = token_dict['token']['ud_pos']
    top_max_k_dicts_lst = token_dict['predictions']['models'][
        config.TARGET_MODEL_NAME
    ]
    top1_token = token_dict['predictions']['models'][config.TARGET_MODEL_NAME][
        0
    ]
    search_in_top_k = [
        {'rank': idx + 1, 'data': d}
        for idx, d in enumerate(top_max_k_dicts_lst)
        if masked_token_str.lower().strip() == d['token_str'].lower().strip()
    ]

    # print({k:v for k,v in token_dict["predictions"].items() if k != "models"})
    # print(token_dict["token"])
    # exit()
    confusion_matrix[masked_token_str.lower().strip()][
        top1_token['token_str'].lower().strip()
    ] += 1
    if top1_token['token_str'].lower().strip() != masked_token_str.lower().strip():
        print(
            f'token_str mismatch {top1_token["token_str"]} != {masked_token_str}'
        )
        print(masked_sentence_str)
        # input()
    for k in [1, 5, 10, 25, 50, 100]:
        if search_in_top_k and search_in_top_k[0]['rank'] <= k:
            # predicted_token_str = prediction['data']['token_str']
            top_k_count['total'][k] += 1
            top_k_count[token_ud_pos][k] += 1
            top_k_count[cefr_level][k] += 1

    top_k_count['total']['count'] += 1
    top_k_count[token_ud_pos]['count'] += 1
    top_k_count[cefr_level]['count'] += 1

    return total_n_tokens, n_skipped_tokens


def calculate_proportions(top_k_count):
    """Calculate the proportions for top_k_count."""
    top_k_proportions = {}
    for dict_id, top_k_count_dict in top_k_count.items():
        total_n_tokens_in_category = top_k_count_dict['count']
        top_k_proportions[dict_id] = {
            k: v / total_n_tokens_in_category
            for k, v in top_k_count_dict.items()
        }
    return top_k_proportions


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as outf:
        json.dump(data, outf)


def process_files_in_directory(
    config, top_k_count, confusion_matrix, total_n_tokens, n_skipped_tokens
):
    """Process all files in the target model directory."""
    for filepath in pathlib.Path(config.TARGET_MODEL_FOLDERPATH).iterdir():
        inputfp = filepath.as_posix()
        print(f'Processing file: {inputfp}')
        text_dicts_dict = load_text_dicts(inputfp)
        for text_id in text_dicts_dict:
            cefr_level = text_dicts_dict[text_id]['text_metadata'][
                config.CEFR_COLUMN
            ]
            # text = text_dicts_dict[text_id]['text']

            tokens_lst = [
                token_dict['token']['token_str']
                for token_dict in text_dicts_dict[text_id]['tokens']
            ]
            for token_dict in text_dicts_dict[text_id]['tokens']:
                total_n_tokens, n_skipped_tokens = process_token_predictions(
                    tokens_lst,
                    token_dict,
                    config,
                    top_k_count,
                    confusion_matrix,
                    cefr_level,
                    total_n_tokens,
                    n_skipped_tokens,
                )
        del text_dicts_dict

    return total_n_tokens, n_skipped_tokens


def main(config):
    """
    Calculate top_k_count, proportions and confusion confusion_matrix_
    """
    top_k_count = collections.defaultdict(lambda: collections.defaultdict(int))
    confusion_matrix = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    total_n_tokens = 0
    n_skipped_tokens = 0

    total_n_tokens, n_skipped_tokens = process_files_in_directory(
        config, top_k_count, confusion_matrix, total_n_tokens, n_skipped_tokens
    )

    top_k_count_fp = (
        f'./results/top_k_count_{config.DATASET}_{config.SPLIT}_{config.TARGET_MODEL_NAME}.json'
        if config.SPLIT
        else f'./results/top_k_count_{config.DATASET}_{config.TARGET_MODEL_NAME}.json'
    )
    save_json(top_k_count, top_k_count_fp)

    top_k_proportions = calculate_proportions(top_k_count)
    top_k_proportions_fp = (
        (
            f'./results/top_k_proportions_{config.DATASET}'
            f'_{config.SPLIT}_{config.TARGET_MODEL_NAME}.json'
        )
        if config.SPLIT
        else f'./results/top_k_proportions_{config.DATASET}_{config.TARGET_MODEL_NAME}.json'
    )
    save_json(top_k_proportions, top_k_proportions_fp)

    confusion_matrix_fp = (
        (
            f'./results/confusion_matrix_{config.DATASET}'
            f'_{config.SPLIT}_{config.TARGET_MODEL_NAME}.json'
        )
        if config.SPLIT
        else f'./results/confusion_matrix_{config.DATASET}_{config.TARGET_MODEL_NAME}.json'
    )
    save_json(confusion_matrix, confusion_matrix_fp)


if __name__ == '__main__':
    CEFR_COLUMNS = {'CELVA': 'CECRL', 'EFCAMDAT': 'cefr'}
    DATASET = 'CELVA'
    user_config = UserConfig(
        target_model_name='bert-base-uncased-finetuned-cleaned-efcamdat__all.txt', #'bert-base-uncased', #'roberta-base',
        dataset=DATASET,
        cefr_column=CEFR_COLUMNS[DATASET],  # "cefr",
        split='',
        data_folderpath=(
            '/home/berstearns/projects/language-learning-modelling/'
            'selva-agreement-metrics/selva-agreement-clients/'
            'poetry-client/outputs/'
        ),
        max_k=100,
    )
    user_config.save_run(folderpath='./run_configs')
    main(user_config)
