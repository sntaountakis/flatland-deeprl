import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DDQNetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, n_actions):
        super(DDQNetwork, self).__init__()
        # Q Network
        self.l1 = nn.Linear(input_dims, l1_dims)
        self.l2 = nn.Linear(l1_dims, l2_dims)
        self.l3 = nn.Linear(l2_dims, n_actions)

    def forward(self, obs):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        
        return q 


class DDDQNetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, n_actions):
        super(DDDQNetwork, self).__init__()
        # Advantage Network
        self.al1 = nn.Linear(input_dims, l1_dims)
        self.al2 = nn.Linear(l1_dims, l2_dims)
        self.al3 = nn.Linear(l2_dims, n_actions)

        # Value Network
        self.vl1 = nn.Linear(input_dims, l1_dims)
        self.vl2 = nn.Linear(l1_dims, l2_dims)
        self.vl3 = nn.Linear(l2_dims, 1)

    def forward(self, state):
        v = F.relu(self.vl1(state))
        v = F.relu(self.vl2(v))
        v = self.vl3(v)
        
        a = F.relu(self.al1(state))
        a = F.relu(self.al2(a))
        a = self.al3(a)

        # Dualing Network Architecture for Deep RL (Wang et. al 2015) 
        q = v + (a - a.mean())
        return q


class LSTMDDDQNetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, n_layers, n_actions):
        super(LSTMDDDQNetwork, self).__init__()
        self.n_layers = n_layers
        self.l1_dims = l1_dims
        
        # Advantage Network
        #self.al1 = nn.LSTM(input_dims, l1_dims, self.n_layers, batch_first=True)
        self.al2 = nn.Linear(l1_dims, l2_dims)
        self.al3 = nn.Linear(l2_dims, n_actions)
        self.a_hidden = None

        # Value Network
        self.vl1 = nn.LSTM(input_dims, l1_dims, self.n_layers, batch_first=True)
        self.vl2 = nn.Linear(l1_dims, l2_dims)
        self.vl3 = nn.Linear(l2_dims, 1)
        self.v_hidden = None

    def init_hidden(self, batch_size):
        a_hidden = (T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'),
                        T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'))
        v_hidden = (T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'),
                        T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'))
        
        return v_hidden, a_hidden

    def forward(self, state, v_hidden=None, a_hidden=None):
        # state.shape -> (batch_size, seq_len, obs_len)
        #if self.a_hidden is None or state.shape[0] != self.a_hidden[0].shape[1]:
        #    self.init_hidden(state.shape[0])
        training_flag = False
        if v_hidden is None:
            training_flag = True
            v_hidden, a_hidden = self.init_hidden(state.shape[0])

        #hidden = (hidden[0].detach(), hidden[1].detach())
        #self.v_hidden = (self.v_hidden[0].detach(), self.v_hidden[1].detach())
        self.vl1.flatten_parameters()
        out, v_hidden = self.vl1(state, v_hidden)
        #v, self.v_hidden = self.vl1(state, self.v_hidden)
        out = out[:, -1, :] #v.shape -> (batch_size, hidden_size)
        v = F.relu(self.vl2(out))
        v = self.vl3(v)
        
        #self.a_hidden = (self.a_hidden[0].detach(), self.a_hidden[1].detach())
        #self.al1.flatten_parameters()
        #a, a_hidden = self.al1(state, a_hidden)
        #a = a[:, -1, :] #v.shape -> (batch_size, hidden_size)
        a = F.relu(self.al2(out))
        a = self.al3(a)

        # Dualing Network Architecture for Deep RL (Wang et. al 2015) 
        q = v + (a - a.mean())
        if not training_flag:
            return q, v_hidden, a_hidden
        return q
'''
class LSTMDDDQNetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, n_layers, n_actions):
        super(LSTMDDDQNetwork, self).__init__()
        self.n_layers = n_layers
        self.l1_dims = l1_dims
        
        self.l1 = nn.LSTM(input_dims, l1_dims, self.n_layers, batch_first=True)
        self.hidden = None
        
        # Advantage Network
        self.al2 = nn.Linear(l1_dims, l2_dims)
        self.al3 = nn.Linear(l2_dims, n_actions)

        # Value Network
        self.vl2 = nn.Linear(l1_dims, l2_dims)
        self.vl3 = nn.Linear(l2_dims, 1)


    def init_hidden(self, batch_size):
        self.hidden = (T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'),
                        T.zeros(self.n_layers, batch_size, self.l1_dims).to('cuda:0'))

    def forward(self, state):
        # state.shape -> (batch_size, seq_len, obs_len)
        if self.hidden is None or state.shape[0] != self.hidden[0].shape[1]:
            self.init_hidden(state.shape[0])

        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.l1.flatten_parameters()
        out, self.hidden = self.l1(state, self.hidden)
        out = out[:, -1, :] #v.shape -> (batch_size, hidden_size)
        
        a = F.relu(self.al2(out))
        a = self.al3(a)

        v = F.relu(self.vl2(out))
        v = self.vl3(v)
        
        q = v + (a - a.mean())
        return q
'''