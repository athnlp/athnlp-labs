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

