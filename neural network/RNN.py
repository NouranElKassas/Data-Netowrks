import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        
        # define model layers
        self.dropout = dropout
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.vocab_size = vocab_size
        
        # define model layers
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)
        # dropout layer
        #self.dropout_layer = nn.Dropout(dropout)
        # linear layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        
        # get batch size as the first dimension of inputs
        batch_size = nn_input.size(0)
        
        # embeddings and LSTM
        embeds = self.embedding(nn_input)
        lstm_output, hidden = self.lstm(embeds, hidden)
        
        # stack up LSTM outputs 
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layers
        lstm_output = self.fc(lstm_output)
        
        # reshape into (batch_size, seq_length, output_size)
        lstm_output = lstm_output.view(batch_size, -1, self.output_size)
        
        # get last batch
        output = lstm_output[:, -1]
        # return one batch of output word scores and the hidden state
        
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_rnn(RNN, train_on_gpu)


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    
    # perform backpropagation and optimization
    
    # creating new variable for the hidden state, otherwise
    # it'd backprop through the entire history
    hidden = tuple([each.data for each in hidden])
    
    # zero accumulated gradients
    rnn.zero_grad()
    
    # get the output from the model
    output, hidden = rnn(inp, hidden)
    
    # calaculate the loss and perform backprop
    loss = criterion(output.squeeze(), target)
    loss.backward()
    
    # perform clipping and optimization
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()
    
    # calculating average loss over the batch
    avg_loss = loss.item()
    # return the loss over a batch and the hidden state produced by our model
    return avg_loss, hidden

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
