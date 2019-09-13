import argparse
import math
import time

import torch

from athnlp.readers.lm_corpus import Corpus

parser = argparse.ArgumentParser(description='RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/lm',
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument("--model_path", type=str, default='models/lm/default.pt',
                    help='Path where to store the trained language model.')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=10,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument("--sentence_compl", action='store_true')


def evaluate(model, criterion, eval_batch_size, corpus, data_source):
    """
    Evaluates the performance of the model according to the specified criterion  on the provided data source

    :param model: RNN language model
    :param criterion: criterion to be evaluated
    :param eval_batch_size: batch size (you can assume 1 for simplicity)
    :param corpus: instance of the reference corpus
    :param data_source: reference data for evaluation
    :return: the average score evaluated using the specified criterion
    """
    pass


def train(model, criterion, corpus, train_data, lr, bptt, epoch):
    """
    Trains the specified language model by minimising the provided criterion using as the training data. It trains the
    model for a given number of epoch with a fixed learning rate.

    :param model: RNN language model
    :param criterion: LM loss function
    :param corpus: Reference corpus
    :param train_data: training data for the LM task
    :param lr: SGD learning rate
    :param bptt: Sequence length
    :param epoch: Number of training epochs
    :return: Average training loss
    """
    pass


def main(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data)

    # training mode selected
    # Trains the model and then runs the evaluation on the test set
    if not args.sentence_compl:
        eval_batch_size = 1
        ###############################################################################
        # Load your train, valid and test data
        ###############################################################################
        # TODO: data loading
        train_data = None
        val_data = None
        test_data = None

        ###############################################################################
        # Build the model
        ###############################################################################
        # TODO: model definition and loss definition
        model = None
        criterion = None

        # Loop over epochs.
        lr = args.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train(model, criterion, corpus, train_data, lr, args.bptt, epoch)
                val_loss = evaluate(model, criterion, eval_batch_size, corpus, val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.model_path, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss

                # HINT: when the loss is not decreasing anymore on the validation set can you think to any method
                # to prevent the model from overfitting?
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(args.model_path, 'rb') as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.rnn.flatten_parameters()

        # Run on test data.
        test_loss = evaluate(model, criterion, eval_batch_size, corpus, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
    else:
        # we enabled the sentence completition mode

        # we first load the model
        # Load the best saved model.
        with open(args.model_path, 'rb') as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.rnn.flatten_parameters()

        ###############################################################################
        # Use the pretrained LM at inference time
        ###############################################################################
        # TODO: Sentence completion solution


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
