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
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        """
        Initialises the parameters of the RNN model.

        N.B. This is optional because you may want to use the default PyTorch weight initialisation
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        """
        Forward pass of the RNN language model. Useful information about how to use
        an RNNCell can be found in the PyTorch documentation:
        https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        https://pytorch.org/docs/stable/nn.html#torch.nn.GRU

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
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
