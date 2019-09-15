import json
import logging
from collections import namedtuple
from typing import Dict, List, Tuple
from typing import Optional

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter
from overrides import overrides
from pytorch_transformers.tokenization_bert import whitespace_tokenize, BertTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
cls_token='[CLS]'
sep_token='[SEP]'
sequence_a_segment_id=0
sequence_b_segment_id=1
cls_token_segment_id=0


@DatasetReader.register("bert_squad")
class BertSquadReader(DatasetReader):

    def __init__(self,
                 max_sequence_length: int,
                 doc_stride: int,
                 question_length_limit: int,
                 lazy: bool = False,
                 version_2: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: WordTokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {}
        self._tokenizer = tokenizer or WordTokenizer()
        self._version_2 = version_2
        self.max_sequence_length = max_sequence_length
        self.doc_stride= doc_stride
        self.question_length_limit = question_length_limit

    def _read(self, file_path: str):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(file_path, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if self._version_2:
                        is_impossible = qa["is_impossible"]
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                    query_tokens = self._tokenizer.tokenize(question_text)

                    if self.question_length_limit is not None and len(query_tokens) > self.question_length_limit:
                        query_tokens = query_tokens[0:self.question_length_limit]

                    tok_to_orig_index = []
                    orig_to_tok_index = []
                    all_doc_tokens = []
                    for (i, token) in enumerate(doc_tokens):
                        orig_to_tok_index.append(len(all_doc_tokens))
                        sub_tokens = self._tokenizer.tokenize(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_doc_tokens.append(sub_token)

                    tok_start_position = None
                    tok_end_position = None
                    if is_impossible:
                        tok_start_position = -1
                        tok_end_position = -1
                    else:
                        tok_start_position = orig_to_tok_index[start_position]
                        if end_position < len(doc_tokens) - 1:
                            tok_end_position = orig_to_tok_index[end_position + 1] - 1
                        else:
                            tok_end_position = len(all_doc_tokens) - 1
                        (tok_start_position, tok_end_position) = _improve_answer_span(
                            all_doc_tokens, tok_start_position, tok_end_position, self._tokenizer,
                            orig_answer_text)

                    # The -3 accounts for [CLS], [SEP] and [SEP]
                    max_tokens_for_doc = self.max_sequence_length - len(query_tokens) - 3

                    # We can have documents that are longer than the maximum sequence length.
                    # To deal with this we do a sliding window approach, where we take chunks
                    # of the up to our max length with a stride of `doc_stride`.
                    _DocSpan = namedtuple(  # pylint: disable=invalid-name
                        "DocSpan", ["start", "length"])
                    doc_spans = []
                    start_offset = 0
                    while start_offset < len(all_doc_tokens):
                        length = len(all_doc_tokens) - start_offset
                        if length > max_tokens_for_doc:
                            length = max_tokens_for_doc
                        doc_spans.append(_DocSpan(start=start_offset, length=length))
                        if start_offset + length == len(all_doc_tokens):
                            break
                        start_offset += min(length, self.doc_stride)

                    for (doc_span_index, doc_span) in enumerate(doc_spans):
                        tokens = []
                        token_to_orig_map = {}
                        token_is_max_context = {}

                        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                        # Original TF implem also keep the classification token (set to 0) (not sure why...)
                        # p_mask = []

                        # CLS token at the beginning
                        # tokens.append(Token(cls_token))
                        # p_mask.append(0)
                        cls_index = 0

                        # Query
                        for token in query_tokens:
                            tokens.append(token)
                            # p_mask.append(1)

                        # SEP token
                        tokens.append(Token(sep_token))
                        # p_mask.append(1)

                        # Paragraph
                        paragraph_start_id = len(tokens)
                        for i in range(doc_span.length):
                            split_token_index = doc_span.start + i
                            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                                   split_token_index)
                            token_is_max_context[len(tokens)] = is_max_context
                            tokens.append(all_doc_tokens[split_token_index])
                            # p_mask.append(0)
                        paragraph_len = doc_span.length

                        # SEP token
                        tokens.append(Token(sep_token))
                        # p_mask.append(1)

                        span_is_impossible = is_impossible
                        start_position = None
                        end_position = None
                        if not span_is_impossible:
                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            out_of_span = False
                            if not (tok_start_position >= doc_start and
                                    tok_end_position <= doc_end):
                                out_of_span = True
                            if out_of_span:
                                span_is_impossible = True
                            else:
                                # we offset by 2 to account for the [CLS] and [SEP] tokens (before/after the question)
                                # at the beginning of the sequence. NOTE: we don't add the [CLS] token, as it will get
                                # added later in the indexer process, therefore start_position will be off by 1.
                                doc_offset = len(query_tokens) + 2
                                start_position = tok_start_position - doc_start + doc_offset
                                end_position = tok_end_position - doc_start + doc_offset

                        if span_is_impossible:
                            start_position = cls_index
                            end_position = cls_index

                        passage_offsets = []
                        token_idx = 0

                        for token in doc_tokens:
                            passage_offsets.append((token_idx, token_idx + len(token)))
                            token_idx += len(token)

                        instance = self.text_to_instance(
                            qas_id=qas_id,
                            question_text=question_text,
                            passage_tokens=tokens[paragraph_start_id: paragraph_len],
                            bert_tokens=tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            answer_texts=[answer["text"] for answer in qa["answers"]],
                            passage_offsets=passage_offsets,
                            passage_text=paragraph["context"]
                        )

                        yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         qas_id: str,
                         question_text: str,
                         bert_tokens: List[Token],
                         passage_tokens: List[Token],
                         orig_answer_text: str,
                         start_position: int,
                         end_position: int,
                         answer_texts: List[str],
                         passage_offsets: List[Tuple[int, int]],
                         passage_text: str) -> Optional[Instance]:
        fields: Dict[str, Field] = {}
        tokens_field = TextField(bert_tokens, self._token_indexers)
        fields['tokens'] = tokens_field

        fields['span_start'] = IndexField(start_position, tokens_field)
        fields['span_end'] = IndexField(end_position, tokens_field)
        metadata = {
            'question_text': question_text,
            'qas_id': qas_id,
            'token_offsets': passage_offsets,
            'original_passage': passage_text
        }

        if answer_texts:
            metadata['answer_texts'] = answer_texts

        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(map(lambda x: x.text, tokenizer.tokenize(orig_answer_text)))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(map(lambda x: x.text, doc_tokens[new_start:(new_end + 1)]))
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


@WordSplitter.register("bert-basic-wordpiece")
class BertBasicWordSplitter(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """
    def __init__(self,
                 pretrained_model: str,
                 do_lower_case: bool = True,
                 never_split: Optional[List[str]] = None) -> None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lower_case)

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(text) for text in self.bert_tokenizer.tokenize(sentence)]


