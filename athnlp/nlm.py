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


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, batch_size, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    num_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, num_batches * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


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
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            # We multiply by the number of examples in the batch in order
            # to get the total loss and not the average (which is what
            # by default PyTorch Cross-entropy loss is doing behind
            # the scenes for us)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


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
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        # TODO: run model forward pass obtaining '(output, hidden)'
        output, hidden = None, None
        # TODO: compute loss using the defined criterion obtaining 'loss'.
        loss = None
        # TODO: compute backpropagation calling the backward pass

        # TODO (optional): implement gradient clipping to prevent
        # the exploding gradient problem in RNNs / LSTMs
        # check the PyTorch function `clip_grad_norm`

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


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
        train_data = batchify(corpus.train, args.batch_size, device)
        val_data = batchify(corpus.valid, eval_batch_size, device)
        test_data = batchify(corpus.test, eval_batch_size, device)

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
