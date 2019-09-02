from athnlp.readers.sequence_dictionary import SequenceDictionary


class Sequence(object):

    def __init__(self, dictionary: SequenceDictionary, x, y, nr):
        self.x = x
        self.y = y
        self.nr = nr
        self.dictionary = dictionary

    def size(self):
        """Returns the size of the sequence."""
        return len(self.x)

    def __len__(self):
        return len(self.x)

    def copy_sequence(self):
        """Performs a deep copy of the sequence"""
        s = Sequence(self.dictionary, self.x[:], self.y[:], self.nr)
        return s

    def update_from_sequence(self, new_y):
        """Returns a new sequence equal to the previous but with y set to newy"""
        s = Sequence(self.dictionary, self.x, new_y, self.nr)
        return s

    def __str__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%s/%s " % (self.dictionary.x_dict.get_label_name(xi),
                               self.dictionary.y_dict.get_label_name(yi))
        return rep

    def __repr__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%s/%s " % (self.dictionary.x_dict.get_label_name(xi),
                               self.dictionary.y_dict.get_label_name(yi))
        return rep
