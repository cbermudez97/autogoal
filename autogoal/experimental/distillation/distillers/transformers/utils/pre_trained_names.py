from typing import Dict, Type, Union
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


DISTIL_PREFIX = "distil"
SQUAD_SUFFIX = "distilled-squad"

KNOW_PAIRS: Dict[str, str] = {
    "bert-base-uncased": "distilbert-base-uncased",
    "bert-base-cased": "distilbert-base-cased",
    "gpt2": "distilgpt2",
    "bert-base-german-dbmdz-cased": "distilbert-base-german-cased",
    "bert-base-multilingual-cased": "distilbert-base-multilingual-cased",
    "roberta-base": "distilroberta-base",
}
KNOW_PAIRS_SQUAD: Dict[str, str] = {
    "bert-base-uncased": "distilbert-base-uncased-distilled-squad",
    "bert-base-cased": "distilbert-base-cased-distilled-squad",
}
KNOW_PAIRS_MODELS: Dict[str, Type[PreTrainedModel]] = {
    "distilbert-base-uncased": DistilBertModel,
    "distilbert-base-cased": DistilBertModel,
    "distilgpt2": DistilBertModel,
    "distilbert-base-german-cased": DistilBertModel,
    "distilbert-base-multilingual-cased": DistilBertModel,
    "distilbert-base-uncased-distilled-squad": DistilBertModel,
    "distilbert-base-cased-distilled-squad": DistilBertModel,
    "distilroberta-base": DistilBertModel,
}
KNOW_PAIRS_TOKENIZERS: Dict[str, Type[PreTrainedTokenizer]] = {
    "distilbert-base-uncased": DistilBertTokenizer,
    "distilbert-base-cased": DistilBertTokenizer,
    "distilgpt2": DistilBertTokenizer,
    "distilbert-base-german-cased": DistilBertTokenizer,
    "distilbert-base-multilingual-cased": DistilBertTokenizer,
    "distilbert-base-uncased-distilled-squad": DistilBertTokenizer,
    "distilbert-base-cased-distilled-squad": DistilBertTokenizer,
    "distilroberta-base": DistilBertTokenizer,
}


def get_pre_trained_name(name: str, use_squad=False) -> Union[str, None]:
    pre_trained_name = None
    if use_squad:
        pre_trained_name = KNOW_PAIRS_SQUAD.get(name, None)
    if not pre_trained_name:
        pre_trained_name = KNOW_PAIRS.get(name, None)
    return pre_trained_name


def get_pre_trained_model(
    name: str, local_files_only=False
) -> Union[PreTrainedModel, None]:
    pre_trained_model_cls: Type[PreTrainedModel] = KNOW_PAIRS_MODELS.get(name, None)
    if pre_trained_model_cls:
        return pre_trained_model_cls.from_pretrained(
            name, local_files_only=local_files_only
        )
    return None


def get_pre_trained_tokenizer(
    name: str, local_files_only=False
) -> Union[PreTrainedTokenizer, None]:
    pre_trained_tokenizer_cls: Type[PreTrainedTokenizer] = KNOW_PAIRS_TOKENIZERS.get(
        name, None
    )
    if pre_trained_tokenizer_cls:
        return pre_trained_tokenizer_cls.from_pretrained(
            name, local_files_only=local_files_only
        )
    return None


def build_pre_trained_name(name: str, use_squad=False) -> str:
    pre_trained_name = DISTIL_PREFIX + name
    if use_squad:
        pre_trained_name += SQUAD_SUFFIX
    return pre_trained_name
