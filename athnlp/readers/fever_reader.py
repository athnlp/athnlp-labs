import json
import logging
from typing import Iterable, Dict, List

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import MetadataField, TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("feverlite")
class FEVERLiteDatasetReader(DatasetReader):
    def __init__(self,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading FEVER instances from {}".format(file_path))
        with open(file_path,"r") as file:
            for line in file:
                json_line = json.loads(line)
                yield self.text_to_instance(**json_line)

    def text_to_instance(self, claim:str, evidence:List[str], label:str=None) -> Instance:
        # Evidence in the dataset is a list of sentences. We can concatenate these into just one long string
        # Extension Exercise: Can you make a new dataset reader and model that handles them individually?
        evidence = " ".join(set(evidence))

        # Tokenize the claim and evidence
        claim_tokens = self._claim_tokenizer.tokenize(claim)
        evidence_tokens = self._wiki_tokenizer.tokenize(evidence)

        instance_meta = {"claim_tokens": claim_tokens,
                         "evidence_tokens": evidence_tokens }

        instance_dict = {"claim": TextField(claim_tokens, self._token_indexers),
                         "evidence": TextField(evidence_tokens, self._token_indexers),
                         "metadata": MetadataField(instance_meta)
                         }

        if label is not None:
            instance_dict["label"] = LabelField(label)

        return Instance(instance_dict)

