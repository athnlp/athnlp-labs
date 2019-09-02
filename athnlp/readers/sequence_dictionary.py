from athnlp.readers.label_dictionary import LabelDictionary


class SequenceDictionary:

    def __init__(self):
        self.x_dict = LabelDictionary()
        self.y_dict = LabelDictionary()
