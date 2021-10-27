from typing import Dict, Union


DISTIL_PREFIX = "distil"
KNOW_PAIRS: Dict[str, str] = {
    "bert-base-uncased": "distilbert-base-uncased",
    "bert-base-cased": "distilbert-base-cased",
    "gpt2": "distilgpt2",
    "bert-base-german-dbmdz-cased": "distilbert-base-german-cased",
    "bert-base-multilingual-cased": "distilbert-base-multilingual-cased",
}

KNOW_PAIRS_SQUAD: Dict[str, str] = {
    "bert-base-uncased": "distilbert-base-uncased-distilled-squad",
    "bert-base-cased": "distilbert-base-cased-distilled-squad",
}


def get_pre_trained_name(name: str, use_squad=False) -> Union[str, None]:
    pre_trained_name = None
    if use_squad:
        pre_trained_name = KNOW_PAIRS_SQUAD.get(name, None)
    if not pre_trained_name:
        pre_trained_name = KNOW_PAIRS.get(name, None)
    return pre_trained_name


def build_pre_trained_name(name: str, use_squad=False) -> str:
    pre_trained_name = "distil" + name
    if use_squad:
        pre_trained_name += "distilled-squad"
    return pre_trained_name
