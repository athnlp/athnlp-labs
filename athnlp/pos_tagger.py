from argparse import ArgumentParser

from athnlp.models.pos_tagger_perceptron import Perceptron
from athnlp.models.pos_tagger_structured_perceptron import StructuredPerceptron
from athnlp.readers.brown_pos_corpus import BrownPosTag

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument('-n', "--num_epochs", type=int, default=10)
    argparse.add_argument('-l', "--print_lines", type=int, default=5000)
    argparse.add_argument('-a', "--averaged", type=bool, default=False)
    argparse.add_argument('-m', "--model_path", type=str, required=True)
    argparse.add_argument("--predict", action='store_true')
    argparse.add_argument("--dataset", choices=["brown"], default="brown")
    argparse.add_argument("--feature_type", choices=["unigrams", "bigrams"], default="unigrams")
    argparse.add_argument("-t","--perceptron_type", choices=["simple", "structured"], default="simple")
    # structured perceptron parameters
    argparse.add_argument("--use_beam", action='store_true')
    argparse.add_argument("--beam_size", type=int, default=-1) # -1 means do brute_force
    argparse.add_argument("--beam_width", type=float, default=0.0)
    # dataset reader parameters
    argparse.add_argument("--max_sent_len", type=int, default=15)
    argparse.add_argument("--num_train_sents", type=int, default=10000)
    argparse.add_argument("--num_dev_sents", type=int, default=1000)
    argparse.add_argument("--num_test_sents", type=int, default=1000)

    args = argparse.parse_args()
    # load dataset
    if args.dataset == "brown":
        dataset = BrownPosTag(max_sent_len=args.max_sent_len,
                              num_train_sents=args.num_train_sents,
                              num_dev_sents=args.num_dev_sents,
                              num_test_sents=args.num_test_sents)
    else:
        raise Exception("Unsupported dataset type.")

    if args.perceptron_type == 'simple':
        tagger = Perceptron(args.num_epochs,
                            args.averaged,
                            args.print_lines,
                            args.feature_type)
    elif args.perceptron_type == 'structured':
        tagger = StructuredPerceptron(args.num_epochs,
                                      args.averaged,
                                      args.print_lines,
                                      args.feature_type,
                                      args.use_beam,
                                      args.beam_size,
                                      args.beam_width)
    else:
        raise Exception("Unsupported perceptron type.")

    tagger.compute_features(args.feature_type, dataset)

    if args.predict:
        print("Loading model from file: {}".format(args.model_path))
        tagger.load_model(args.model_path)
        num_mistakes_total, num_labels_total, predicted_sequences = tagger.predict(dataset.dictionary, dataset.test)
        print("Accuracy: %f" % (tagger.accuracy(num_mistakes_total, num_labels_total)))
        for i in range(len(predicted_sequences)):
            print("%i:\t%s" % (i, predicted_sequences[i]))
    else:
        print("Training model {}\n# epochs: {}\n# training instances = {}\n# dev instances = {}".format(
            args.perceptron_type,
            args.num_epochs,
            len(dataset.train),
            len(dataset.dev)
        ))
        tagger.train(dataset.dictionary, dataset.train, dataset.dev, len(dataset.dictionary.y_dict))

        print("Saving trained model to file: {}".format(args.model_path))
        tagger.save_model(args.model_path)
