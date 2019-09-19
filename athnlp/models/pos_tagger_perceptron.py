import numpy as np
from overrides import overrides

from athnlp.models.abstract_model import Model
from athnlp.readers.sequence import Sequence
from athnlp.readers.sequence_dictionary import SequenceDictionary


class Perceptron(Model):
    def __init__(self, num_epochs, averaged, print_lines, feature_type):
        super().__init__()
        self.num_epochs = num_epochs
        self.averaged = averaged
        self.features = None
        self.feature_type = feature_type
        self.print_lines = print_lines

    @overrides
    def train(self, dictionary: SequenceDictionary, train_set: [Sequence], dev_set: [Sequence], num_classes, seed=1):
        self.parameters = np.zeros((len(self.features), num_classes))
        params_per_epoch = []

        for epoch in range(self.num_epochs):
            print("\nEpoch ", epoch)
            num_labels_total = 0
            num_mistakes_total = 0

            # use seed to generate permutation
            np.random.seed(seed)
            train_shuffled = np.random.permutation(train_set)

            # change the seed so next epoch we don't get the same permutation
            seed += 1

            for i in range(len(train_shuffled)):
                example = train_shuffled[i]
                num_labels, num_mistakes, _ = self.perceptron_activation(example, update=True)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
                if i % self.print_lines == 0:
                    print("Trained %i examples Train accuracy: %f" %
                          (i, self.accuracy(num_mistakes_total, num_labels_total)))
            if self.averaged:
                params_per_epoch.append(self.parameters.copy())
            # compute training accuracy per epoch
            print("Epoch: %i Train accuracy: %f" % (epoch, self.accuracy(num_mistakes_total, num_labels_total)))

            # compute dev accuracy
            dev_num_mistakes_total, dev_num_labels_total, _ = self.predict(dictionary, dev_set)
            print("Epoch: %i Dev accuracy: %f" % (epoch, self.accuracy(dev_num_mistakes_total, dev_num_labels_total)))

        if self.averaged:
            new_w = 0
            for old_w in params_per_epoch:
                new_w += old_w
            new_w /= len(params_per_epoch)
            self.parameters = new_w

    def perceptron_activation(self, sequence: Sequence, update: bool):
        num_mistakes = 0
        seq_length = len(sequence.y)
        y_hats = []
        # compute a label for each word and update parameters individually
        for x, y in zip(self.vectorize(sequence.x), sequence.y):
            scores = np.dot(x, self.parameters)
            y_hat = np.argmax(scores, axis=0).transpose()
            y_hats.append(y_hat)

            if y != y_hat:
                num_mistakes += 1
                if update:
                    # increase gold features
                    self.parameters[:, y] += x
                    # decrease predicted features
                    self.parameters[:, y_hat] -= x
        return seq_length, num_mistakes, y_hats

    def vectorize(self, sequence):
        vec_seq = []
        for x in sequence:
            if self.feature_type == 'unigrams':
                # trivial case: the id of the word is also the unigram id already stored in the dictionary
                vec = np.zeros(len(self.features))
                vec[x] = 1
                vec_seq.append(vec)
            elif self.feature_type == 'bigrams':
                raise Exception("Unsupported vectorization operation")
        return vec_seq

    @overrides
    def predict(self, dictionary: SequenceDictionary, data: [Sequence]):
        num_labels_total = 0
        num_mistakes_total = 0
        predicted_sequences = []
        for i in range(len(data)):
            example = data[i]
            num_labels, num_mistakes, predictions = self.perceptron_activation(example, update=False)
            predicted_sequences.append(Sequence(dictionary, example.x, predictions, i))
            num_labels_total += num_labels
            num_mistakes_total += num_mistakes
        return num_mistakes_total, num_labels_total, predicted_sequences

    def compute_features(self, feature_type: str, dataset):
        if feature_type == 'unigrams':
            self.features = dataset.dictionary.x_dict
        elif feature_type == 'bigrams':
            self.features = dataset.dictionary.x_dict
            self.features.append(self.compute_ngrams(dataset, n=2))
        else:
            raise Exception("Unsupported feature type")

    def compute_ngrams(self, dataset, n):
        # TODO: complete
        return dataset.dictionary.x_dict