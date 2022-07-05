import torch as T
import torch.nn as nn 
import torch.nn.functional as F

class COMACritic(nn.Module):
    def __init__(self, input_dims, l1_dims, n_actions):
        super(COMACritic, self).__init__()
        self.l1 = nn.Linear(input_dims, l1_dims)
        self.l2 = nn.Linear(l1_dims, l1_dims)
        self.l3 = nn.Linear(l1_dims, n_actions)

    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class COMAActor(nn.Module):
    def __init__(self, input_dims, l1_dims):
        super(COMAActor, self).__init__()
        self.l1 = nn.Linear(input_dims, l1_dims)
        self.l2 = nn.Linear(l1_dims, l1_dims)
        self.l3 = nn.Linear(l1_dims, 5)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)

        probs = F.softmax(a, dim=-1)
        dist = T.distributions.Categorical(probs)

        return dist
    
class PPONetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, out_dims):
        super(PPONetwork, self).__init__()

        # Actor
        self.pl1 = nn.Linear(input_dims, l1_dims)
        self.pl2 = nn.Linear(l1_dims, l2_dims)
        self.pl3 = nn.Linear(l2_dims, out_dims)
 
        # Critic
        self.vl1 = nn.Linear(input_dims, l1_dims)
        self.vl2 = nn.Linear(l1_dims, l2_dims)
        self.vl3 = nn.Linear(l2_dims, 1)

        nn.init.orthogonal_(self.pl1.weight)
        nn.init.orthogonal_(self.pl2.weight)
        nn.init.orthogonal_(self.pl3.weight)

        nn.init.orthogonal_(self.vl1.weight)
        nn.init.orthogonal_(self.vl2.weight)
        nn.init.orthogonal_(self.vl3.weight)

    def forward(self, obs):
        # obs.shape -> (batch_size, input_dims*n_agents)
        # obs[:,:input_dims] -> (batch_size, input_dims)
        policy = T.tanh(self.pl1(obs))
        policy = T.tanh(self.pl2(policy))
        policy = self.pl3(policy)

        value = T.tanh(self.vl1(obs))
        value = T.tanh(self.vl2(value))
        value = self.vl3(value)
        
        probs = F.softmax(policy, dim=-1)
        dists = T.distributions.Categorical(probs)

        return dists, value.squeeze(-1)
