import torch as T
import random
import operator
from collections import namedtuple

#Value Transition
V_Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
P_Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'mask'))
P2_Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'logprob')) 
TransData = namedtuple('TransData', ('priority', 'prob', 'weight', 'indx'))


class ReplayBuffer():
    def __init__(self, params, policy=False):
        self.policy = policy
        self.max_size = params.buffer_size
        self.memory = []
        self.position = 0
        self.device = T.device("cuda:0")
        #if alg == "DDQN":
        self.transition = P2_Transition if policy else V_Transition

    def store(self, state, action, reward, next_state, done, mask=None, logprob=None):
        if len(self.memory) < self.max_size:
            self.memory.append(None)

        if self.policy:
            #self.memory[self.position] = self.transition(state, action, logprob, reward, 
            #                                        next_state, done, mask)
            self.memory[self.position] = self.transition(state, action, reward, next_state, done, logprob)
        else:
            self.memory[self.position] = self.transition(state, action, reward, 
                                                    next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = self.transition(*zip(*batch))
        
        states = T.tensor(batch.state).float().to(device=self.device).detach()
        actions = T.tensor(batch.action).to(device=self.device).detach()
        rewards = T.tensor(batch.reward).float().to(device=self.device).detach()
        next_states = T.tensor(batch.next_state).float().to(device=self.device).detach()
        dones = T.tensor(batch.done, dtype=T.bool).to(self.device).detach()
        logprobs = None
        if self.policy:
            dones = dones.int()
            #masks = T.tensor(batch.mask, dtype=T.bool).int().to(self.device).detach()
            logprobs = T.tensor(batch.logprob).float().to(self.device).detach()
            #return states, actions, logprobs, rewards, next_states, dones, masks
        
        return states, actions, rewards, next_states, dones, logprobs 
    
    def __len__(self):
        return len(self.memory)
    

class PriorityReplay():
    def __init__(self, params, policy=None):
        self.max_size = params.buffer_size
        self.policy = policy
        self.transition = P_Transition if policy else V_Transition
        self.device = T.device("cuda:0")
        self.position = 0
        traj_data = []
        for i in range(self.max_size):
            traj_data.append(TransData(0, 0, 0, i))

        self.memory = {key: self.transition for key in range(self.max_size)}
        self.mem_data = {key: data for key, data in zip(range(self.max_size), traj_data)}
        self.trajectories_stored = 0 
        self.alpha = params.alpha      # Affects the importance of the priority 
        self.beta = params.beta
        self.alpha_decay = params.alpha_decay
        self.beta_rise = params.beta_rise
        self.priority_cumsum = 0 
        self.max_priority = 1   #When adding a new trajectory it needs to get assigned to max priority
        self.max_weight = 1

    def update_priorities(self, delta, indexes):
        '''
            Update priorities and weights of the 
            transitions that were used on learning
        '''
        N = min(self.trajectories_stored, self.max_size)
    
        # Update our cummulative sum of priorities to the new priorities
        #self.max_weight = max(self.mem_data.values(), key=operator.itemgetter(2)).weight
        for i, indx in enumerate(indexes):   
            new_priority = delta[i]
            if not new_priority:
                new_priority = 0.00000001
            
            if new_priority > self.max_priority:
                self.max_priority = new_priority
                
            new_weight = ((N*new_priority)**(-self.beta))/self.max_weight
            if new_weight > self.max_weight:
                self.max_weight = new_weight

            # Update the cummulative priority sum to the new priorities
            self.priority_cumsum -= self.mem_data[indx].priority**self.alpha
            self.priority_cumsum += new_priority**self.alpha

            # Calculate the new probabilities
            new_prob = new_priority ** self.alpha / self.priority_cumsum
            if new_prob < 0:
                print('debug')
            
            # Update the selected batch of trajectory data to the new priority, weight, prob
            self.mem_data[indx] = TransData(new_priority, new_prob, new_weight, indx)

    def update_params(self):
        self.alpha *= self.alpha_decay 
        self.beta *= self.beta_rise 
        if self.beta > 1:
            self.beta = 1
        
        #Priorities, probs, weights get updated with the new parameters
        N = min(self.trajectories_stored, self.max_size)
        self.priority_cumsum = 0
        for i in range(N):
            self.priority_cumsum += self.mem_data[i].priority**self.alpha
        
        for i in range(N):
            prob = self.mem_data[i].priority**self.alpha / self.priority_cumsum
            weight = ((1/N)*(1/self.mem_data[i].priority))**self.beta
            if weight > self.max_weight:
                self.max_weight = weight
            self.mem_data[i] = TransData(self.mem_data[i].priority, prob, weight, i)
        
    def store(self, state, action, reward, next_state, done, mask=None):
        self.trajectories_stored += 1

        if self.trajectories_stored > self.max_size:
            #This trajectory will get replaced and we have to remove it's priority from the cummulative sum of priorities
            self.priority_cumsum -= self.mem_data[self.position].priority**self.alpha
            #TODO: Do the update of the max priorities if the prev_max got removed
            # For Computational efficiency only
            if self.mem_data[self.position].priority == self.max_priority:
                tmp = self.mem_data[self.position]
                self.mem_data[self.position] = TransData(0, tmp.prob, tmp.weight, tmp.indx)
                #self.mem_data[self.position].priority = 0
                self.max_priority = max(self.mem_data.values(), key=operator.itemgetter(0)).priority
            if self.mem_data[self.position].weight == self.max_weight:
                tmp = self.mem_data[self.position]
                self.mem_data[self.position] = TransData(0, tmp.prob, 0, tmp.indx)
                self.max_weight = max(self.mem_data.values(), key=operator.itemgetter(2)).weight  
        

        self.priority_cumsum += self.max_priority**self.alpha 
        
        prob = self.max_priority ** self.alpha / (self.priority_cumsum)

        if mask:
            self.memory[self.position] = self.transition(state, action, reward, next_state, done, mask)
        else:
            self.memory[self.position] = self.transition(state, action, reward, next_state, done)
        self.mem_data[self.position] = TransData(self.max_priority, prob, self.max_weight, self.position)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        dataset = list(self.mem_data.values())
        batch_data = random.choices(self.mem_data, [mdata.prob for mdata in dataset], k=batch_size)
        batch = []
        for data in batch_data: 
            batch.append(self.memory.get(data.indx))
        #random.sample(self.memory, batch_size)
        batch = self.transition(*zip(*batch))
        batch_data = TransData(*zip(*batch_data))

        states = T.tensor(batch.state).float().to(device=self.device).detach()
        actions = T.tensor(batch.action).to(device=self.device).detach()
        rewards = T.tensor(batch.reward).float().to(device=self.device).detach()
        next_states = T.tensor(batch.next_state).float().to(device=self.device).detach()
        dones = T.tensor(batch.done, dtype=T.bool).to(self.device).detach()
        if self.policy:
            # TEMPORARY MPAKALIKO FIX, dones HOLDS state_values
            dones = T.tensor(batch.done).float().to(self.device).detach()
            masks = T.tensor(batch.mask, dtype=T.bool).int().to(self.device).detach()    
            return states, actions, rewards, next_states, dones, masks, batch_data
        return states, actions, rewards, next_states, dones, batch_data
    
    def __len__(self):
        return len(self.memory)


class EpisodeBuffer():
    def __init__(self):
        self.batch = []
        self.pos = 0
        self.device = T.device('cuda:0')
        
    def store(self, state, action, reward, n_state, done, mask): 
        self.batch.append(None)
        self.batch[self.pos] = P_Transition(state, action, reward, n_state, done, mask) 
        self.pos += 1

    def sample(self):
        

        batch = P_Transition(*zip(*self.batch))
        states = T.tensor(batch.state).float().to(self.device).detach()
        actions = T.tensor(batch.action).to(self.device).detach()
        rewards = T.tensor(batch.reward).unsqueeze(-1).to(self.device).detach()
        dones = T.tensor(batch.done).int().unsqueeze(-1).to(self.device).detach()
        next_states = T.tensor(batch.next_state).float().to(self.device).detach()
        masks = T.tensor(batch.mask).int().unsqueeze(-1).to(self.device).detach()

        #last_states = T.tensor(batch.n_state)[-1:].float().to(self.device).detach()
        
        #critic_states = self.shape_critic_input(states, actions, parm)
        #critic_last_states = self.shape_critic_input(last_states, None, parm)
        #next_critic_states = self.shape_critic_input(next_states, actions, parm)
        

        #states = states.view(parm.batch_size*parm.n_agents, -1)
        #critic_states = critic_states.view(parm.batch_size*parm.n_agents, -1)
        #actions = actions.view(-1, 1)

        return states, actions, rewards, next_states, dones, masks 
    
    def shape_critic_input(self, states, actions, parm):
        batch_size = states.shape[0]
        if actions is None:
            actions = T.zeros(batch_size, parm.n_agents).long().to(self.device)
        
        ret = T.zeros(batch_size, parm.n_agents, parm.critic_dims).to(self.device)
        mixed_obs = states.view(-1, states.shape[1]*states.shape[2])
        for h in range(parm.n_agents):
            other_actions = T.cat((actions[:, :h], actions[:, h+1:]), dim=1).unsqueeze(2)
            actions_onehot = T.zeros(batch_size, parm.n_agents-1, parm.n_actions).to(self.device)
            actions_onehot.scatter_(2, other_actions, 1)
            actions_onehot = actions_onehot.view(-1, parm.n_actions*(parm.n_agents-1))
            id = T.zeros((batch_size, 1), dtype=int).to(self.device).fill_(h)
            id_onehot = T.zeros(batch_size, parm.n_agents).to(self.device)
            id_onehot.scatter_(1, id, 1)     
            ret[:, h, :] = T.cat((mixed_obs, actions_onehot, id_onehot), dim=1)

        return ret
    
    def reset(self):
        self.batch.clear()
        self.pos = 0
    
    def __len__(self):
        return len(self.batch)
