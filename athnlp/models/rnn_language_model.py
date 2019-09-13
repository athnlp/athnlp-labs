import torch.nn as nn


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        """
        Initialises the parameters of the RNN Language Model

        :param rnn_type: type of RNN cell
        :param ntoken: number of tokens in the vocabulary
        :param ninp: Dimensionality of the input vector
        :param nhid: Hidden size of the RNN cell
        :param nlayers: Number of layers of the RNN cell
        :param dropout: Dropout value applied to the RNN cell connections
        """
        super(RNNModel, self).__init__()

    def init_weights(self):
        """
        Initialises the parameters of the RNN model.

        N.B. This is optional because you may want to use the default PyTorch weight initialisation
        """
        pass

    def forward(self, input, hidden):
        """
        Forward pass of the RNN language model. You are free to implement it as you wish so we won't provide you with
        any constraints on the shape of the input tensors.

        :param input: input features
        :param hidden: previous hidden state of the RNN language model
        :return: output of the model
        """
        pass

    def init_hidden(self, bsz):
        """
        Returns the initial hidden state of the RNN language model. It is a function that should be called before
        unrolling the RNN decoder.

        :param bsz: batch size
        :return: first hidden state of the RNN language model
        """
        pass