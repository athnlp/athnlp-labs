from typing import Optional, Dict, List, Any

import allennlp
import torch
from allennlp.nn.util import get_text_field_mask
from torch import nn
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("fever")
class FEVERTextClassificationModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 final_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:

        super().__init__(vocab,regularizer)

        # Model components
        self._embedder = text_field_embedder
        self._feed_forward = final_feedforward

        # For accuracy and loss for training/evaluation of model
        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()

        # Initialize weights
        initializer(self)


    def forward(self,
                claim: Dict[str, torch.LongTensor],
                evidence: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        claim : Dict[str, torch.LongTensor]
            From a ``TextField``
            The LongTensor Shape is typically ``(batch_size, sent_length)`
        evidence : Dict[str, torch.LongTensor]
            From a ``TextField``
            The LongTensor Shape is typically ``(batch_size, sent_length)`
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the claim and
            evidence sentences with 'claim_tokens' and 'premise_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        raise NotImplementedError("Compute label logits (for supported and refuted) for the given Claim and Evidence input")

        # TODO - Uncomment the code below

        #label_logits = # TODO compute label logits for input
        #label_probs = F.softmax(label_logits, dim=-1)

        #output_dict = {"label_logits": label_logits,
        #               "label_probs": label_probs}

        #if label is not None:
        #    loss = self._loss(label_logits, label.long().view(-1))
        #    self._accuracy(label_logits, label)
        #    output_dict["loss"] = loss

        #if metadata is not None:
        #    output_dict["claim_tokens"] = [x["claim_tokens"] for x in metadata]
        #    output_dict["evidence_tokens"] = [x["evidence_tokens"] for x in metadata]

        #return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }
