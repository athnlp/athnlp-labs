import sys
import warnings


class LabelDictionary(dict):
    """This class implements a dictionary of labels. Labels are mapped to
    integers, as it is more efficient to retrieve the label name from its
    integer representation, and vice-versa."""

    def __init__(self, label_names=[]):
        self.names = []
        for name in label_names:
            self.add(name)

    def add(self, name):
        if name in self:
            # warnings.warn('Ignoring duplicated label ' + name)
            label_id = self[name]
        else:
            label_id = len(self.names)
            self[name] = label_id
            self.names.append(name)
        return label_id

    def get_label_name(self, label_id):
        return self.names[label_id]

    def get_label_id(self, name):
        return self[name]
