import torch as T
import torch.optim as optim
import torch.nn.functional as F
from memory.replay_buffer import PriorityReplay, ReplayBuffer
from models.value_networks import DDQNetwork, DDDQNetwork, LSTMDDDQNetwork
from collections import deque
import statistics as stats
import numpy as np
import copy
import random
import os

class DQNPolicy():
    def __init__(self, input_dims, n_actions, params, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        self.input_dims = input_dims
        self.l1_dims = 125
        self.l2_dims = 125
        self.n_actions = n_actions
        self.n_agents = 5
        self.epsilon = 0
        if not evaluation_mode:
            self.l1_dims = params.hidden_layer
            self.l2_dims = params.hidden_layer
            self.lrate = params.lrate
            self.gamma = params.gamma
            self.epsilon = params.epsilon
            self.eps_min = params.eps_min
            self.eps_decay = params.eps_decay
            self.buffer_size = params.buffer_size
            self.batch_size = params.batch_size
            self.learn_step = 0
            self.learn_freq = params.learn_freq
            self.priority = params.priority
            self.update_target_steps = 0 # Not used
            
        if params.use_gpu:
            self.device = T.device("cuda:0")
        else:
            self.device = T.device("cpu")
        
        self.policy_net = DDQNetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_actions).to(device=self.device)

        if not evaluation_mode:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lrate)
            self.memory = PriorityReplay(params) if params.priority else ReplayBuffer(params)
 
    def act(self, state, eval=False):
        self.policy_net.eval()
        state = T.tensor([state]).float().to(self.device)
        rnum = random.random()
        if (rnum > self.epsilon) or eval:
            with T.no_grad():
                action = self.policy_net(state).max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        self.policy_net.train()
        return action

    def step(self, state, action, reward, next_state, done, agent=None): 
        # Store Transition
        self.memory.store(state, action, reward, next_state, done)
        self.learn_step += 1
        
        # Learn
        if self.learn_step%(self.learn_freq) == 0:
            self.learn()
            self.learn_step = 0
        

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, data = self.memory.sample(self.batch_size)

        qvals = self.policy_net(states).gather(1, actions.unsqueeze(1))

        next_qvals = self.policy_net(next_states).max(1)[0].detach()
        next_qvals[dones] = 0

        target_qvals = (rewards + (self.gamma*next_qvals)).unsqueeze(1)

        loss = F.mse_loss(qvals, target_qvals.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update Batch priorities in the Prioritized Replay Buffers
        if self.priority:
            delta = abs(qvals - target_qvals).squeeze().detach().cpu().tolist()
            self.memory.update_priorities(delta, list(data.indx))

    def update_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon*self.eps_decay)

    def save(self, path):
        T.save(self.policy_net, path)
        
    def load(self, path):
        self.policy_net = T.load(path)


class DDQNPolicy():
    def __init__(self, input_dims, n_actions, params, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        self.input_dims = input_dims
        self.l1_dims = 125
        self.l2_dims = 125
        self.n_actions = n_actions
        self.n_agents = 5
        self.epsilon = 0
        if not evaluation_mode:
            self.l1_dims = params.hidden_layer
            self.l2_dims = params.hidden_layer
            self.lrate = params.lrate
            self.gamma = params.gamma
            self.epsilon = params.epsilon
            self.eps_min = params.eps_min
            self.eps_decay = params.eps_decay
            self.buffer_size = params.buffer_size
            self.batch_size = params.batch_size
            self.learn_step = 0
            self.learn_freq = params.learn_freq
            self.update_target_steps = 0
            self.update_target_freq = params.update_target_freq
            self.priority = params.priority
            self.stored_experiences = 0
            
        if params.use_gpu:
            self.device = T.device("cuda:0")
        else:
            self.device = T.device("cpu")
        
        self.policy_net = DDQNetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_actions).to(device=self.device)

        if not evaluation_mode:
            self.target_net = copy.deepcopy(self.policy_net)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lrate)
            self.memory = PriorityReplay(params) if params.priority else ReplayBuffer(params)
 
    def act(self, state, eval=False):
        self.policy_net.eval()
        state = T.tensor([state]).float().to(self.device)
        rnum = random.random()
        if (rnum > self.epsilon) or eval:
            with T.no_grad():
                action = self.policy_net(state).max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        self.policy_net.train()
        return action

    def step(self, state, action, reward, next_state, done, agent=None): 
        # Store Transition
        self.memory.store(state, action, reward, next_state, done)
        self.stored_experiences += 1
        self.learn_step += 1
        
        # Learn
        if self.learn_step%(self.learn_freq) == 0:
            self.learn()
            self.learn_step = 0
        
        # Increase self.update_target_steps at the end of each episode
        # in the main function. Update the target every ~8 episodes
        if self.update_target_steps%(self.update_target_freq) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.update_target_steps = 1
            
    def learn(self):
        if self.stored_experiences < self.batch_size:
            return

        states, actions, rewards, next_states, dones, data = self.memory.sample(self.batch_size)

        qvals = self.policy_net(states).gather(1, actions.unsqueeze(1))

        next_qvals_policy = self.policy_net(next_states).max(1)[0].detach()
        next_qvals_target = self.target_net(next_states).max(1)[0].detach()
        next_qvals = T.min(next_qvals_policy, next_qvals_target)
        next_qvals[dones] = 0

        target_qvals = (rewards + (self.gamma*next_qvals)).unsqueeze(1)

        loss = F.mse_loss(qvals, target_qvals.detach())
        if self.priority:
            weights = list(data.weight)
            with T.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *=weight
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.priority:
            delta = abs(qvals - target_qvals).squeeze().detach().cpu().tolist()
            self.memory.update_priorities(delta, list(data.indx))

    def update_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon*self.eps_decay)

    def save(self, path):
        T.save(self.policy_net, path)
        

    def load(self, path):
        self.policy_net = T.load(path)


class DDQNsoftPolicy(DDQNPolicy):
    def __init__(self, input_dims, n_actions, params, evaluation_mode):
        super().__init__(input_dims, n_actions, params, evaluation_mode=evaluation_mode)
        self.tau = 0.01 # See optimal tau
        self.debug_priority = 0

    def step(self, state, action, reward, next_state, done, h): 
        # Store Transition
        self.memory.store(state, action, reward, next_state, done)
        self.stored_experiences += 1
        self.learn_step += 1
        
        # Learn
        if self.learn_step%(self.learn_freq) == 0:
            self.learn()
            self.learn_step = 0
                

    def learn(self):
        if self.stored_experiences < self.batch_size:
            return

        states, actions, rewards, next_states, dones, data = self.memory.sample(self.batch_size)
        

        qvals = self.policy_net(states).gather(1, actions.unsqueeze(1))

        next_qvals = self.target_net(next_states).max(1)[0].detach()
        next_qvals[dones] = 0

        target_qvals = (rewards + (self.gamma*next_qvals)).unsqueeze(1)

        #loss = F.mse_loss(qvals, target_qvals.detach())
        self.debug_priority = abs(qvals - target_qvals.detach()).mean().item()
        loss = ((qvals - target_qvals.detach()) ** 2)
        #if self.priority:
        #    weights = T.tensor(data.weight).float().detach().to('cuda:0')
        #    loss *= weights.unsqueeze(-1)
            #weights = list(data.weight)
            #with T.no_grad():
            #    weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            #loss *=weight
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net)
        if self.priority:
            delta = abs(qvals - target_qvals.detach()).squeeze().detach().cpu().tolist()
            self.memory.update_priorities(delta, list(data.indx))
        
        
    def soft_update(self, policy_net, target_net):
        for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class DDDQNPolicy(DDQNsoftPolicy):
    def __init__(self, input_dims, n_actions, params, evaluation_mode=False):
        super().__init__(input_dims, n_actions, params, evaluation_mode)        
        self.policy_net = DDDQNetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_actions).to(device=self.device)

        if not evaluation_mode:
            self.target_net = copy.deepcopy(self.policy_net)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lrate)
            self.memory = PriorityReplay(params) if params.priority else ReplayBuffer(params)
        

class DRQNPolicy(DDQNsoftPolicy):
    def __init__(self, input_dims, n_actions, params, evaluation_mode=False):
        super().__init__(input_dims, n_actions, params, evaluation_mode)        
        self.n_layers = params.n_layers
        self.history_length = params.history_length   
        self.priority = params.priority
        self.policy_net = LSTMDDDQNetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_layers, self.n_actions).to(device=self.device)
        self.n_agents = 5

        if not evaluation_mode:
            self.target_net = copy.deepcopy(self.policy_net)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lrate)
            self.memory = PriorityReplay(params) if params.priority else ReplayBuffer(params)
        self.history = [deque(maxlen=self.history_length) for _ in range(self.n_agents)]

    def act(self, state, v_hidden, a_hidden, eval=False):
        self.policy_net.eval()
        state = T.tensor([state]).float().to(self.device)
        if self.evaluation_mode:
            state = state.unsqueeze(1)
        
        rnum = random.random()
        if (rnum > self.epsilon):
            with T.no_grad():
                q, v_hidden, a_hidden = self.policy_net(state, v_hidden, a_hidden)
                action = q.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        self.policy_net.train()
        return action, v_hidden, a_hidden
    
    def stack_states(self, state, agent):
        if not self.history[agent]:
            #self.history[agent].append(state)
            #return np.expand_dims(state, axis=0)
            for _ in range(self.history_length - 1):
                self.history[agent].append(np.zeros_like(state))

        self.history[agent].append(state)
        state = np.stack([s for s in self.history[agent]], axis=0)
        return state
