# pylint: disable=no-self-use
from typing import List, Dict
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer, WordpieceIndexer


@TokenIndexer.register("bert-squad-indexer")
class BertSquadIndexer(PretrainedBertIndexer):
    """
    TokenIndexer closely based on AllenNLP's WordpieceIndexer; the only major difference is that we assume that
    basic and then wordpiece tokenization have already taken place when reading the SQuAD dataset
    (this follows the original methodology of hugginface). The reason we do that is so that start_position and
    end_position are correctly offset due to the extra wordpiece tokens introduced.
    NOTE: We are unnecesarily checking for len(tokens) > max_pieces, as we have already split the paragraphs
    when reading the dataset. The corresponding code below should never be triggered.
    """
    def __init__(self,
                 pretrained_model: str) -> None:
        super().__init__(pretrained_model)

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (token.text.lower()
                if self._do_lowercase and token.text not in self._never_lowercase
                else token.text
                for token in tokens)

        # Create nested sequence of wordpieces
        nested_wordpiece_tokens = _get_nested_wordpiece_tokens([token for token in text])

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [[self.vocab[wordpiece] for wordpiece in token]
                               for token in nested_wordpiece_tokens]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = _get_token_type_ids(flat_wordpiece_ids, self._separator_ids)

        # The code below will (possibly) pack the wordpiece sequence into multiple sub-sequences by using a sliding
        # window `window_length` that overlaps with previous windows according to the `stride`. Suppose we have
        # the following sentence: "I went to the store to buy some milk". Then a sliding window of length 4 and
        # stride of length 2 will split them up into:

        # "[I went to the] [to the store to] [store to buy some] [buy some milk [PAD]]".

        # This is to ensure that the model has context of as much of the sentence as possible to get accurate
        # embeddings. Finally, the sequences will be padded with any start/end piece ids, e.g.,

        # "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ...".

        # The embedder should then be able to split this token sequence by the window length,
        # pass them through the model, and recombine them.

        # Specify the stride to be half of `self.max_pieces`, minus any additional start/end wordpieces
        window_length = self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        stride = window_length // 2

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = len(self._start_piece_ids) if self.use_starting_offsets else len(self._start_piece_ids) - 1

        # Count amount of wordpieces accumulated
        pieces_accumulated = 0
        for token in token_wordpiece_ids:
            # Truncate the sequence if specified, which depends on where the offsets are
            next_offset = 1 if self.use_starting_offsets else 0
            if self._truncate_long_sequences and offset + len(token) - 1 >= window_length + next_offset:
                break

            # For initial offsets, the current value of ``offset`` is the start of
            # the current wordpiece, so add it to ``offsets`` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            # For final offsets, the current value of ``offset`` is the end of
            # the previous wordpiece, so increment it and then add it to ``offsets``.
            else:
                offset += len(token)
                offsets.append(offset)

            pieces_accumulated += len(token)

        if len(flat_wordpiece_ids) <= window_length:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)]
            token_type_ids = self._extend(flat_token_type_ids)
        elif self._truncate_long_sequences:
            self._warn_about_truncation(tokens)
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[:pieces_accumulated])]
            token_type_ids = self._extend(flat_token_type_ids[:pieces_accumulated])
        else:
            # Create a sliding window of wordpieces of length `max_pieces` that advances by `stride` steps and
            # add start/end wordpieces to each window
            # TODO: this currently does not respect word boundaries, so words may be cut in half between windows
            # However, this would increase complexity, as sequences would need to be padded/unpadded in the middle
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[i:i + window_length])
                                 for i in range(0, len(flat_wordpiece_ids), stride)]

            token_type_windows = [self._extend(flat_token_type_ids[i:i + window_length])
                                  for i in range(0, len(flat_token_type_ids), stride)]

            # Check for overlap in the last window. Throw it away if it is redundant.
            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window):]:
                wordpiece_windows = wordpiece_windows[:-1]
                token_type_windows = token_type_windows[:-1]

            token_type_ids = [token_type for window in token_type_windows for token_type in window]

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]


        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {index_name: wordpiece_ids,
                f"{index_name}-offsets": offsets,
                f"{index_name}-type-ids": token_type_ids,
                "mask": mask}


def _get_token_type_ids(wordpiece_ids: List[int],
                        separator_ids: List[int]) -> List[int]:
    num_wordpieces = len(wordpiece_ids)
    token_type_ids: List[int] = []
    type_id = 0
    cursor = 0
    while cursor < num_wordpieces:
        # check length
        if num_wordpieces - cursor < len(separator_ids):
            token_type_ids.extend(type_id
                                  for _ in range(num_wordpieces - cursor))
            cursor += num_wordpieces - cursor
        # check content
        # when it is a separator
        elif all(wordpiece_ids[cursor + index] == separator_id
                 for index, separator_id in enumerate(separator_ids)):
            token_type_ids.extend(type_id for _ in separator_ids)
            type_id += 1
            cursor += len(separator_ids)
        # when it is not
        else:
            cursor += 1
            token_type_ids.append(type_id)
    return token_type_ids


def _get_nested_wordpiece_tokens(flat_wordpiece_tokens: List[str]):
    nested_worpiece_tokens = []
    nested = []
    for wordpiece in flat_wordpiece_tokens:
        if wordpiece.startswith("##"):
            nested.append(wordpiece)
        else:
            nested = [wordpiece]
            nested_worpiece_tokens.append(nested)
    return nested_worpiece_tokens
