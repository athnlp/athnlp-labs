import abc

import numpy as np

from athnlp.readers.sequence import Sequence
from athnlp.readers.sequence_dictionary import SequenceDictionary


class Model(abc.ABC):

    def __init__(self):
        self.parameters = None

    @abc.abstractmethod
    def train(self, dictionary: SequenceDictionary, train_set: [Sequence], dev_set: [Sequence], num_classes, seed=1):
        pass

    @abc.abstractmethod
    def predict(self, dictionary: SequenceDictionary, data: [Sequence]):
        pass

    @staticmethod
    def accuracy(mistakes, labels):
        return 1.0 - mistakes / labels

    def save_model(self, model_path):
        with open(model_path, 'w') as out_file:
            # write dimensions
            out_file.write("%i %i\n" % (self.parameters.shape[0], self.parameters.shape[1]))
            # write parameters in easy-to-read format (for educational purposes only)
            for p_id, vec in enumerate(self.parameters):
                out_file.write("%i\t%s\n" % (p_id, '\t'.join([str(val) for val in self.parameters[p_id]])))

    def load_model(self, model_path):
        with open(model_path, 'r') as in_file:
            dims = tuple([int(dim) for dim in in_file.readline().split()])
            self.parameters = np.empty(dims)
            for line in in_file:
                toks = line.strip().split("\t")
                p_id = int(toks[0])
                vec = [float(tok) for tok in toks[1:]]
                self.parameters[p_id] = vec
