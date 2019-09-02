from nltk.corpus import brown
from athnlp.readers.sequence_dictionary import SequenceDictionary
from athnlp.readers.sequence import Sequence


class BrownPosTag:

    def __init__(
            self,
            max_sent_len: int = 15,
            num_train_sents=10000,
            num_dev_sents=1000,
            num_test_sents=1000,
            mapping_file="athnlp/readers/en-brown.map"):

        self.train = []
        self.dev = []
        self.test = []
        self.dictionary = SequenceDictionary()

        # Build mapping of postags
        self.mapping = {}
        if mapping_file is not None:
            for line in open(mapping_file):
                coarse, fine = line.strip().split("\t")
                self.mapping[coarse.lower()] = fine.lower()

        # Initialize noun to be tag zero so that it the default tag
        self.dictionary.y_dict.add("noun")

        # Preprocess dataset splits
        sents = brown.tagged_sents()
        last_id = 0
        self.train, last_id = self.preprocess_split(sents, last_id, num_train_sents, max_sent_len, "train_")
        self.dev, last_id = self.preprocess_split(sents, last_id, num_dev_sents, max_sent_len, prefix_id="dev_")
        self.test, _ = self.preprocess_split(sents, last_id, num_test_sents, max_sent_len, prefix_id="test_")

    def preprocess_split(self, input_dataset, last_id, num_sents, max_sent_len, prefix_id = ""):
        """"Add necessary pre-processing (e.g., convert to universal tagset) to sentences of the dataset."""
        dataset = []
        for sent in input_dataset[last_id:]:
            last_id += 1
            if type(sent) == tuple or len(sent) > max_sent_len or len(sent) <= 1:
                continue
            dataset.append(self.preprocess_sent(sent, prefix_id + str(len(dataset))))
            if len(dataset) == num_sents:
                break

        return dataset, last_id

    def preprocess_sent(self, sent, sent_id):
        """Every word and tag of the sentence gets mapped to a unique id stored in a SequenceDictionary instance."""
        ids_x = []
        ids_y = []
        for word, tag in sent:
            tag = tag.lower()
            if tag not in self.mapping:
                # Add unk tags to mapping dict
                self.mapping[tag] = "noun"
            universal_tag = self.mapping[tag]
            word_id = self.dictionary.x_dict.add(word)
            tag_id = self.dictionary.y_dict.add(universal_tag)
            ids_x.append(word_id)
            ids_y.append(tag_id)
        return Sequence(self.dictionary, ids_x, ids_y, sent_id)


if __name__ == '__main__':
    corpus = BrownPosTag()
    print("vocabulary size: ", len(corpus.dictionary.x_dict))
    print("train/dev/test set length: ", len(corpus.train), len(corpus.dev), len(corpus.test))
    print("First train sentence: ", corpus.train[0])
    print("First dev sentence: ", corpus.dev[0])
    print("First test sentence: ", corpus.test[0])
