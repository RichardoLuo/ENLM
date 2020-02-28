"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, List, Set
import json
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
import numpy as np
from overrides import overrides

from enlm.data import AliasDatabase
from enlm.data.fields import SequentialArrayField

logger = logging.getLogger(__name__)

MAX_PARENTS = 10


def _flatten(nested: Iterable[str]):
    return [x for seq in nested for x in seq]

def _tokenize(iterable: Iterable[str]):
    return [Token(x) for x in iterable]


@DatasetReader.register('enhanced-wikitext')
class EnhancedWikitextReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

    @overrides
    def text_to_instance(self, data: Dict[str, Any]) -> Instance:  # pylint: disable=arguments-differ
        # Flatten and pad tokens
        tokens = data['tokens']
        tokens = [Token(x) for x in tokens]
        fields = {'tokens': TextField(tokens, self._token_indexers)}
        return Instance(fields)


@DatasetReader.register("enhanced-wikitext-entity-nlm")
class EnhancedWikitextEntityNlmReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

    @overrides
    def text_to_instance(self, data: Dict[str, Any]) -> Instance:  # pylint: disable=arguments-differ
        # Flatten and pad tokens
        tokens = data['tokens']
        tokens = [Token(x) for x in tokens]
        fields = {'tokens': TextField(tokens, self._token_indexers)}

        # If annotations are provided, process them into arrays.
        if 'annotations' in data:

            # Initialize arrays and book keeping data structures.
            seen_entities: Set[str] = set()
            entity_types = np.zeros(shape=(len(tokens),))
            entity_ids = np.zeros(shape=(len(tokens),))
            mention_lengths = np.ones(shape=(len(tokens),))

            # Process annotations
            for annotation in data['annotations']:

                seen_entities.add(annotation['id'])
                start, end = annotation['span']
                length = end - start

                for i in range(*annotation['span']):
                    # Note: +1 offset to account for start token.
                    entity_types[i] = 1
                    entity_ids[i] = len(seen_entities)
                    mention_lengths[i] = length
                    length -= 1

            fields['entity_types'] = SequentialArrayField(entity_types, dtype=np.uint8)
            fields['entity_ids'] = SequentialArrayField(entity_ids, dtype=np.int64)
            fields['mention_lengths'] = SequentialArrayField(mention_lengths, dtype=np.int64)

        return Instance(fields)


def normalize_entity_id(raw_entity_id: str) -> str:
    if raw_entity_id[0] == 'T':
        entity_id = '@@DATE@@'
    elif raw_entity_id[0] == 'V':
        entity_id = '@@QUANTITY@@'
    elif raw_entity_id[0] in ['P', 'Q']:
        entity_id = raw_entity_id
    else:
        entity_id = None
    return entity_id


