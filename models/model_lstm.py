import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# define an lstm model to predict the next state given the current state and action
class LSTM_model(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.input_size = config.input_size
        self.config = config
        self.hidden_size = hidden_size
        self.output_size = config.output_size
        if self.config.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        elif self.config.rnn_type == 'RNN':
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=False)

        self.linear_in = nn.Linear(self.input_size, self.hidden_size, )
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)
        self.device = self.config.device
        self.init_hidden()
        self.to(self.device)

        weights = [p for n,p in self.named_parameters() if n !='thalamus']
        self.WU_optimizer = torch.optim.Adam(weights, lr=self.config.WU_lr)

        self.thalamus_size = config.no_of_contexts
        self.register_parameter(name='thalamus',param=torch.nn.Parameter(torch.ones([self.config.seq_size,self.config.batch_size,self.thalamus_size])/self.thalamus_size ,requires_grad = True))
        # self.thalamus.to(torch.device('cuda'))
        ### not sure why above is not working
        self.thalamus.data = self.thalamus.data.to(self.device)
        ### had to do the above to move it...
        self.LU_optimizer = self.get_LU_optimizer()
        if self.config.thalamus_activation_function == 'softmax':
            self.thalamus_activation_function = self.thalamus_activation_function_softmax
        elif self.config.thalamus_activation_function == 'none':
            self.thalamus_activation_function = self.thalamus_activation_function_none
    def get_LU_optimizer(self):
        if self.config.LU_optimizer == 'Adam':
            LU_optimizer = torch.optim.Adam([self.thalamus], lr=self.config.LU_lr, weight_decay= self.config.l2_loss if self.config.l2_loss else 0)
        elif self.config.LU_optimizer == 'SGD':
            LU_optimizer = torch.optim.SGD([self.thalamus], lr=self.config.LU_lr, momentum=self.config.momentum)
        return LU_optimizer
    def thalamus_activation_function_softmax(self, x):
        if self.config.no_of_latents == 1:
            sm = torch.softmax(x/self.config.activation_fxn_temp, dim = -1, ) # note: 0, 1 becomes 0.2689, 0.7311
        elif self.config.no_of_latents > 1:
            latent_size = int(self.config.thalamus_size / self.config.no_of_latents)
            # resize x to be [seq, batch, latent_size, no_of_latents]
            # then softmax along the latent_size dimension
            # then reshape back to [seq, batch, thalamus_size]
            x = x.reshape(x.shape[0], x.shape[1], self.config.no_of_latents, latent_size, )
            sm = torch.softmax(x, dim = 3)
            sm = sm.reshape(x.shape[0],  x.shape[1],self.config.thalamus_size)

        return sm
    def thalamus_activation_function_none(self, x):
        return x 
    def init_hidden(self):
        if self.config.rnn_type == 'LSTM':
            self.hidden = (torch.zeros(1, self.config.batch_size, self.hidden_size).to(self.device),
                       torch.zeros(1, self.config.batch_size, self.hidden_size).to(self.device))
        elif self.config.rnn_type == 'RNN':
            self.hidden = torch.zeros(1, self.config.batch_size, self.hidden_size).to(self.device)

    def forward(self, input, hidden= None, reward = None, thalamic_inputs=None):
        '''
        input is expected to be [seq, batch, feature] and torch type float32
        thalamic inputs can be of shape [thalamus_size]
        or [seq, batch, thalamus_size], the code checks and corrects to match inputs.shape 
        
        Update, too messy to be fixing shapes here for all the different cases,
        so now thalamic_inputs is expected to be [seq, batch, thalamus_size]
        or [1, batch, thalamus_size] and will repeat along seq dimension
        '''
        if thalamic_inputs is None:
            thalamic_inputs = self.thalamus.data
            if thalamic_inputs.shape[0] != input.shape[0]:
                print( 'thalamus shape does not match input shape')
                print( 'will use the last value of thalamus for all seq')
                thalamic_inputs = thalamic_inputs[-1].unsqueeze(0).repeat(input.shape[0],1,1)            
        if thalamic_inputs.shape[0] == 1: # repeat along seq dimension
            thalamic_inputs = thalamic_inputs.repeat(input.shape[0],1,1)
        # thalamic_inputs = (thalamic_inputs)\
            # .reshape(1,self.config.batch_size,self.thalamus_size)\
            #     .repeat(self.config.seq_size,1,1)
        # print('thalamic_inputs.shape', thalamic_inputs.shape)
        # print('input.shape', input.shape)
        # if thalamic_inputs.ndim < input.ndim: # reshape into seq, batch, thalamus_size
            # thalamic_inputs = thalamic_inputs.reshape(-1,1,self.thalamus_size)
        # if thalamic_inputs.shape[0] < input.shape[0]: # repeat along seq dimension
        # if thalamic_inputs.shape[1] < input.shape[1]: # repeat along batch dimension
        #     thalamic_inputs = thalamic_inputs.repeat(1,input.shape[1],1)

        # check if thalamus has changed its dim, if so, redefine it as a parameter, and reattach to optimizer
        if thalamic_inputs.shape != self.thalamus.shape:
            self.thalamus = torch.nn.Parameter(thalamic_inputs, requires_grad = True)
            self.thalamus.data = self.thalamus.data.to(self.device)
            self.LU_optimizer = self.get_LU_optimizer()
        else:
            self.thalamus.data = thalamic_inputs.to(self.device)
        
        try:
            input = torch.cat([input, self.thalamus_activation_function(self.thalamus)], dim=2)
        except:
            print('input.shape', input.shape)
            print('self.thalamus.shape', self.thalamus.shape)
            raise
                
        if reward is not None and self.config.use_reward_feedback:
            reward = reward.unsqueeze(0).unsqueeze(2)
            input = torch.cat([input, reward], dim=2)
        if hidden is None:
            hidden = self.hidden
        
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1,1, figsize=[8,2]); axes.plot(input.squeeze().detach()) ;plt.savefig('./output.jpg')    

        input = self.linear_in(input)
        output, hidden = self.rnn(input, hidden)
        self.hidden = hidden # update the hidden state with the new hidden state, if no hidden state is passed excplicitly to forward, this will be used next call
        output = self.linear_out(output)
        return output, hidden

    def predict(self, input):
        output, hidden = self.forward(input, hidden= self.hidden)
        return output, hidden

    def reset(self):
        self.init_hidden()

