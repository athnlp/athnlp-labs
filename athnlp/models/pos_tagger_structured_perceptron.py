import numpy as np
from overrides import overrides
from athnlp.models.pos_tagger_perceptron import Perceptron
from athnlp.readers.sequence import Sequence
from copy import deepcopy


class StructuredPerceptron(Perceptron):
    def __init__(self, num_epochs, averaged, print_lines, feature_type, use_beam, beam_size, beam_width):
        super().__init__(num_epochs, averaged, print_lines, feature_type)
        self.use_beam = use_beam
        self.beam_size = beam_size
        self.beam_width = beam_width

    @overrides
    def perceptron_activation(self, sequence: Sequence, update: bool):
        num_mistakes = 0
        num_classes = self.parameters.shape[-1]

        partial_hypotheses = [([], 0.0)]
        # self.vectorize: [feat_size] of seq_length elements
        for x in self.vectorize(sequence.x):
            # x: (feat_size, 1),  self.parameters: (feat_size, num_classes), scores: (num_classes, 1)
            scores = np.dot(x, self.parameters)
            current_word_hypotheses = []
            for partial_hypothesis in partial_hypotheses:
                for y_hat in range(num_classes):
                    # partial_hypothesis is a tuple now
                    new_partial_hypothesis = deepcopy(partial_hypothesis)
                    new_partial_hypothesis[0].append(y_hat)
                    current_hypothesis = new_partial_hypothesis[0]
                    current_score = new_partial_hypothesis[1]+scores[y_hat]
                    if not self.use_beam or (current_score >= self.beam_width):
                        # to avoid TypeError: 'tuple' object does not support item assignment
                        # we create a new tuple here
                        new_partial_hypothesis = (current_hypothesis, current_score)
                        current_word_hypotheses.append(new_partial_hypothesis)
            # apply beam search
            if self.use_beam:
                partial_hypotheses = sorted(current_word_hypotheses, key=lambda x: x[1], reverse=True)[:self.beam_size]
            else:
                partial_hypotheses = current_word_hypotheses

        y_hats, max_score = sorted(partial_hypotheses, key=lambda x: x[1], reverse=True)[0]

        for y, y_hat in zip(sequence.y, y_hats):
            if y != y_hat:
                num_mistakes += 1
                if update:
                    # increase gold features
                    self.parameters[:, y] += x
                    # decrease predicted features
                    self.parameters[:, y_hat] -= x
        return num_classes, num_mistakes, y_hats
