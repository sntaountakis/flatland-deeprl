import torch as T
import torch.optim as optim
import torch.nn.functional as F
from models.policy_networks import PPONetwork
from memory.replay_buffer import PriorityReplay, EpisodeBuffer
from reinforcement_learning.policy import Policy
import statistics as stats


class PPOPolicy(Policy):
    def __init__(self, input_dims, n_actions, n_agents, params, evaluation_mode=False):
        self.alg = params.alg
        self.evaluation_mode = evaluation_mode
        self.input_dims = input_dims
        self.l1_dims = 125
        self.l2_dims = 125
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.epsilon = 0 #not used
        self.logprobs = []
        if not evaluation_mode:
            self.l1_dims = params.hidden_layer
            self.l2_dims = params.hidden_layer
            self.lrate = params.lrate
            self.gamma = params.gamma
            self.buffer_size = params.buffer_size
            self.batch_size = params.batch_size
            self.tau = params.tau
            self.clip = params.clip 
            self.c1 = params.c1
            self.c2 = params.c2
            self.lamda = 0.95
            self.epochs = params.epochs
            self.sil_epochs = params.sil_epochs
            self.learn_flag = False
            self.memory = EpisodeBuffer()
        if params.use_gpu:
            self.device = T.device("cuda:0")
        else:
            self.device = T.device("cpu")

        self.net = PPONetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_actions).to(device=self.device)
        
        if not evaluation_mode:    
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrate)
            if params.alg=='SIL':
                self.replay = PriorityReplay(params, policy=True)
 
    def act(self, state, eval=False):
        state = T.tensor([state]).float().to(self.device)
        dist, _ = self.net(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        if eval:
            return action.item()
        else:
            return action.item(), logprob.item()

    def step(self, state, action, reward, next_state, done, mask, logprob=None): 
        # Store Transition
        self.logprobs.append(logprob)
        action = list(action.values())
        reward = list(reward.values())
        done = list(done.values())[:-1]
        self.memory.store(state, action, reward, next_state, done, mask)
        

    def learn(self):
        # Get transitions from episode memory
        states, actions, rewards, next_states, dones, masks = self.memory.sample()

        #compute and Normalize Returns
        returns = self.compute_episode_returns(rewards, dones, next_states, masks)
        returns = (returns - self.global_mean) / (self.global_std + 1e-07)
        if self.alg == "SIL":
            #fill Replay Buffer
            self.fill_replay_buffer(returns)
        
        # Calculate old logprobs
        dist, oldvalue = self.net(states)
        #oldlogprob = dist.log_prob(actions).detach()
        oldlogprob = T.tensor(self.logprobs).to(self.device)
        returns = returns.squeeze(-1)
        masks = masks.squeeze(-1)
        oldlogprob[masks == 0] = 0
        returns[masks == 0] = 0
        oldvalue[masks == 0] = 0

        # train PPO on multiple epochs
        mini_batch = min(len(self.memory), self.batch_size)
        for epoch in range(self.epochs):
            
            # Sample mini-batch
            #indx = random.sample(range(len(self.memory)), mini_batch)
            num_samples = masks.sum()

            # Get current NN estimates  
            dist, value = self.net(states) 
            value[masks == 0] = 0

            # Get current NN log_probs of actions on states
            logprob = dist.log_prob(actions)
            logprob[masks == 0] = 0
            
            # Calculate ratio
            ratio = T.exp(logprob - oldlogprob)
            
            # Calculate advantage 
            advantage = returns - value.detach()
        
            # Calculate Policy loss 
            obj1 = ratio*advantage
            obj2 = T.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)*advantage
            obj = T.min(obj1, obj2).mean()
            
            # Calculate Value loss
            clipped_value = oldvalue.detach() + (value - oldvalue.detach()).clamp(min=-self.clip, max=self.clip)
            
            v_loss1 = F.mse_loss(value, returns)
            v_loss2 = F.mse_loss(clipped_value, returns)
            v_loss = T.max(v_loss1, v_loss2)

            # Calculate entropy 
            entropy = dist.entropy()
            entropy[masks==0] = 0
            entropy = entropy.mean()
            
            # Total loss
            loss = -obj + self.c1*v_loss - self.c2*entropy
            
            # Gradient step 
            self.optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        self.memory.reset()
        self.logprobs.clear()
        return loss.detach().item()
    
    def learn_sil(self):
        for _ in range(self.sil_epochs):
            # Prioritized sampling of batch
            states, actions, returns, next_states, dones, masks, data = self.replay.sample(self.batch_size)
            returns[masks == 0] = 0
    
            # Calculate policy loss
            dist, value = self.net(states)
            value[masks == 0] = 0

            # Calculate log prob of selected actions
            logprobs = dist.log_prob(actions)
            logprobs[masks == 0] = 0

            # Calculate advantage
            tmp_returns = returns.clone()
            returns[returns < value] = 0
            value[value > tmp_returns] = 0
            advantage = (returns - value).detach()
            
            policy_loss = (logprobs*advantage).mean()

            v_loss = F.mse_loss(value, returns)
            
            delta = (returns - value).sum(-1).cpu().tolist()
            indxes = list(data.indx)
            self.replay.update_priorities(delta=delta, indexes=indxes)
            loss = -policy_loss + 0.05*v_loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def compute_episode_returns(self, rewards, dones, next_states, masks):
        _, next_values = self.net(next_states[-1])
        next_values = next_values.unsqueeze(-1).detach()
        ret = T.zeros_like(rewards)
        #acc_rewards = next_values*(1 - dones[-1]) 
        acc_rewards = 0
        for t in reversed(range(rewards.shape[0])):
            acc_rewards = rewards[t] + self.gamma*acc_rewards*(1 - dones[t])
            ret[t] = acc_rewards
        return ret
    
    def build_means_and_stds(self, max_steps):
        ret = []
        acc_rewards = 0
        for i in reversed(range(max_steps-1)):
            acc_rewards = -1 + self.gamma*acc_rewards
            ret.insert(0, acc_rewards)
        
        self.global_mean = stats.mean(ret)
        self.global_std = stats.stdev(ret)


    def fill_replay_buffer(self, returns):
        returns = returns.squeeze(-1).clone().cpu().numpy()
        for indx, tr in enumerate(self.memory.batch):
            self.replay.store(tr.state, tr.action, returns[indx], tr.next_state, tr.done, tr.mask)

    def end_episode(self):
        pass

    def save(self, path):
        T.save(self.net, path)
        

    def load(self, path):
        self.net = T.load(path)


class PPORPolicy(Policy):
    def __init__(self, input_dims, n_actions, n_agents, params, evaluation_mode=False):
        self.alg = params.alg
        self.evaluation_mode = evaluation_mode
        self.input_dims = input_dims
        self.l1_dims = 125
        self.l2_dims = 125
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.epsilon = 0
        
        self.ret_means = []
        self.ret_stds = []
        #self.logprobs = [[] for _ in range(self.n_agents)]
        self.logprobs = []
        if not evaluation_mode:
            self.l1_dims = params.hidden_layer
            self.l2_dims = params.hidden_layer
            self.lrate = params.lrate
            self.gamma = params.gamma
            self.buffer_size = params.buffer_size
            self.batch_size = params.batch_size
            self.tau = params.tau
            self.clip = params.clip 
            self.c1 = params.c1
            self.c2 = params.c2
            self.lamda = 0.95
            self.epochs = params.epochs
            self.sil_epochs = params.sil_epochs
            self.learn_flag = False
            self.memory = EpisodeBuffer()
        if params.use_gpu:
            self.device = T.device("cuda:0")
        else:
            self.device = T.device("cpu")

        self.net = PPONetwork(self.input_dims, self.l1_dims, self.l2_dims, 
                                    self.n_actions).to(device=self.device)
        
        if not evaluation_mode:    
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrate)
            
            self.replay = PriorityReplay(params, policy=True)
            #if params.alg=='SIL':
            #    self.replay = PriorityReplay(params, policy=True)
 
    def act(self, state, eval=False):
        state = T.tensor([state]).float().to(self.device)
        dist, _ = self.net(state)
        action = dist.sample()
        logprob = dist.log_prob(action)
        if eval:
            return action.item()
        else:
            return action.item(), logprob.item()

    def step(self, state, action, reward, next_state, done, mask, logprob=None): 
        # Store Transition
        self.logprobs.append(logprob)
        action = list(action.values())
        reward = list(reward.values())
        done = list(done.values())[:-1]
        self.memory.store(state, action, reward, next_state, done, mask)

    def end_episode(self):
        states, actions, rewards, next_states, dones, masks = self.memory.sample()

        # compute Returns
        returns = self.compute_episode_returns(rewards, dones, next_states, masks)
        returns = (returns - self.global_mean) / (self.global_std + 1e-07)
        #returns = (returns - returns.mean()) / (returns.std() + 1e-07)
        
        # Calculate old logprobs
        dist, oldvalue = self.net(states)
        oldlogprob = dist.log_prob(actions).detach()
        #oldlogprob = T.tensor(self.logprobs).to(self.device)
        #oldlogprobs = self.logprobs
        self.fill_replay_buffer(returns, oldlogprob, oldvalue.detach())
        self.memory.reset()
        self.logprobs.clear()

    def learn(self):
        if self.replay.trajectories_stored < self.batch_size:
            return 0

        # train PPO on multiple epochs
        for epoch in range(self.epochs):
            
            states, actions, returns, oldlogprob, oldvalues, masks, _ = self.replay.sample(self.batch_size)

            oldlogprob[masks == 0] = 0
            returns[masks == 0] = 0
            oldvalues[masks == 0] = 0

            # Get current NN estimates  
            dist, value = self.net(states) 
            value[masks == 0] = 0

            # Get current NN log_probs of actions on states
            logprob = dist.log_prob(actions)
            logprob[masks == 0] = 0

            # Calculate ratio
            ratio = T.exp(logprob - oldlogprob)
            
            # Calculate advantage 
            advantage = returns - value.detach()
            
            # Calculate Policy loss 
            obj1 = ratio*advantage
            obj2 = T.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)*advantage
            obj = T.min(obj1, obj2).mean()
            
            # Calculate Value loss
            clipped_value = oldvalues + (value - oldvalues).clamp(min=-self.clip, max=self.clip)
            v_loss1 = F.mse_loss(value, returns)
            v_loss2 = F.mse_loss(clipped_value, returns)
            v_loss = T.max(v_loss1, v_loss2)

            # Calculate entropy 
            entropy = dist.entropy()
            entropy[masks==0] = 0
            entropy = entropy.mean()
            # Total loss
            loss = -obj + self.c1*v_loss - self.c2*entropy
            
            # Gradient step 
            self.optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
        return loss.detach().item()
    
    def learn_sil(self):
        for _ in range(self.sil_epochs):
            # Prioritized sampling of batch
            states, actions, returns, next_states, dones, masks, data = self.replay.sample(self.batch_size)

            returns[masks == 0] = 0

            #states = T.tensor(batch.state).float().to(self.device).detach()
            #actions = T.tensor(batch.action).to(self.device).detach()
            #returns = T.tensor(batch.reward).to(self.device).detach()
            #masks = T.tensor(batch.mask).int().to(self.device).detach()
    
            # Calculate policy loss
            dist, value = self.net(states)
            value[masks == 0] = 0

            # Calculate log prob of selected actions
            logprobs = dist.log_prob(actions)
            logprobs[masks == 0] = 0

            # Calculate advantage
            tmp_returns = returns.clone()
            returns[returns < value] = 0
            value[value > tmp_returns] = 0
            advantage = (returns - value).detach()
            

            policy_loss = (logprobs*advantage).mean()

            v_loss = F.mse_loss(value, returns)
            
            delta = (returns - value).sum(-1).cpu().tolist()
            indxes = list(data.indx)
            self.replay.update_priorities(delta=delta, indexes=indxes)
            loss = -policy_loss + 0.05*v_loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def compute_episode_returns(self, rewards, dones, next_states, masks):
        _, next_values = self.net(next_states[-1])
        next_values = next_values.unsqueeze(-1).detach()
        ret = T.zeros_like(rewards)
        #acc_rewards = next_values*(1 - dones[-1]) 
        acc_rewards = 0
        for t in reversed(range(rewards.shape[0])):
            acc_rewards = rewards[t] + self.gamma*acc_rewards*(1 - dones[t])
            ret[t] = acc_rewards
        return ret
    
    def build_means_and_stds(self, max_steps):
        ret = []
        acc_rewards = 0
        for i in reversed(range(max_steps-1)):
            acc_rewards = -1 + self.gamma*acc_rewards
            ret.insert(0, acc_rewards)
        
        self.global_mean = stats.mean(ret)
        self.global_std = stats.stdev(ret)

    def fill_replay_buffer(self, returns, logprobs, values):
        returns = returns.squeeze(-1).clone().cpu().numpy()
        logprobs = logprobs.clone().cpu().numpy()
        values = values.clone().cpu().numpy()
        for indx, tr in enumerate(self.memory.batch):
            self.replay.store(tr.state, tr.action, returns[indx], logprobs[indx], values[indx], tr.mask)

    def save(self, path):
        T.save(self.net, path)
        

    def load(self, path):
        self.net = T.load(path)