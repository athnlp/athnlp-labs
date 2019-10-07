from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from overrides import overrides
from pytorch_transformers.modeling_bert import BertModel


@Model.register("qa_bert")
class BertQuestionAnswering(Model):
    """
    A QA model for SQuAD based on the AllenNLP Model ``BertForClassification`` that runs pretrained BERT,
    takes the pooled output, adds a Linear layer on top, and predicts two numbers: start and end span.

    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.
    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.
    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: BertModel,
                 dropout: float = 0.0,
                 index: str = "bert",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, ) -> None:
        super().__init__(vocab, regularizer)
        self._index = index
        self.bert_model = PretrainedBertModel.load(bert_model)
        hidden_size = self.bert_model.config.hidden_size

        ###
        # TODO: check if trainable
        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout)

        self._final_layer = torch.nn.Linear(hidden_size, 2)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        ###
        initializer(self._final_layer)

    def forward(self,  # type: ignore
                metadata: Dict,
                tokens: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        span_start : torch.IntTensor, optional (default = None)
            A tensor of shape (batch_size, 1) which contains the start_position of the answer
            in the passage, or 0 if impossible. This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : torch.IntTensor, optional (default = None)
            A tensor of shape (batch_size, 1) which contains the end_position of the answer
            in the passage, or 0 if impossible. This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        start_probs: torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        end_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        best_span:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        """
        Outputs will be a tuple containing:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. 
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        """
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=input_mask)

        last_layer_output = outputs[0]

        # apply linear layer for every wordpiece token
        logits = self._final_layer(self._dropout(last_layer_output))

        # (batch_size, max_sequence_length, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        question_mask = token_type_ids.clone().float().log() + 1
        # In SQUAD 2.0 there are "impossible questions" 
        # To deal with this, wee need to allow the probability mass to be on the [CLS] token
        # This unmasks [CLS], which is always the first token in the input
        question_mask[:, 0] = 1
        start_probs = masked_softmax(start_logits, mask=question_mask, dim=-1)
        end_probs = masked_softmax(end_logits, mask=question_mask, dim=-1)
        best_span = get_best_span(start_probs, end_probs)

        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_probs": start_probs,
            "end_probs": end_probs,
            "best_span": best_span,
            "passage_question_attention": outputs[-1]
        }

        if span_start is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            span_start.clamp_(0, ignored_index)
            span_end.clamp_(0, ignored_index)

            loss = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss(start_logits, span_start.squeeze(-1))
            end_loss = loss(end_logits, span_end.squeeze(-1))
            self._span_start_accuracy(start_logits, span_start.squeeze(-1))
            self._span_end_accuracy(end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.cat([span_start, span_end], -1))
            output_dict["loss"] = (start_loss + end_loss) / 2

            # TODO: double-check this
            if metadata is not None:
                batch_size = end_logits.shape[0]
                output_dict["metadata"] = metadata
                output_dict["tokens"] = tokens
                output_dict = self.decode(output_dict)

                for i in range(batch_size):
                    best_span_string = output_dict['best_span_str'][i]
                    answer_texts = metadata[i].get('answer_texts', ["[CLS]"])
                    if answer_texts:
                        self._squad_metrics(best_span_string, answer_texts)
                    print("pred: " + best_span_string + " gold:" + str(answer_texts))
                #######

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        best_span_str = []
        batch_size = output_dict["start_probs"].shape[0]
        best_spans = output_dict["best_span"]
        tokens = output_dict["tokens"]

        for i in range(batch_size):
            predicted_span = tuple(best_spans[i].detach().cpu().numpy())
            start_span = predicted_span[0]
            end_span = predicted_span[1]
            pred_answer_tokens = [self.vocab._index_to_token["bert"][token] for token in tokens["bert"][i].tolist()][
                                 start_span: end_span + 1]
            # start_offset = offsets[predicted_span[0]][0]
            # end_offset = offsets[predicted_span[1]][1]
            # best_span_string = passage_str[start_offset:end_offset]
            best_span_str.append(self.wordpiece_to_tokens(pred_answer_tokens))
        output_dict['best_span_str'] = best_span_str

        return output_dict

    @staticmethod
    def wordpiece_to_tokens(wordpiece_tokens):
        buffer = ""

        for w_token in wordpiece_tokens:
            if "##" not in w_token:
                buffer += " " + w_token
            else:
                buffer += w_token.replace("##", "")

        return buffer.strip()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)

        metrics = {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }
        return metrics


class PretrainedBertModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, BertModel] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> BertModel:
        if model_name in cls._cache:
            return PretrainedBertModel._cache[model_name]

        model = BertModel.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model
