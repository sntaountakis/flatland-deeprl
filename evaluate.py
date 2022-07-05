from train import get_env
from torch.functional import norm
from torch.utils.tensorboard import SummaryWriter
from utils.observation_utils import normalize_observation
from reinforcement_learning.ppo_policy import PPOPolicy, PPORPolicy
from reinforcement_learning.extra_extra_policies import PPORPolicy
from reinforcement_learning.value_policy import DDQNPolicy, DDQNsoftPolicy,DDDQNPolicy, DQNPolicy, DRQNPolicy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from utils.misc import get_level
from utils.deadlock_check import check_if_all_blocked
from argparse import Namespace, ArgumentParser
import pandas as pd
import numpy as np
import sys
import os

def main(params):

    value_policies = ['DQN', 'DDQN', 'DDDQN', 'DRQN']
    ppo_policies = ['PPO', 'PPOR' ,'SIL']
    
    # Set env parameters
    env_params =  Namespace(**get_level(difficulty=params.level, seed=params.seed))

    # Setup predictor
    predictor = ShortestPathPredictorForRailEnv(params.tree_depth)

    # Setup observation
    obs_object = TreeObsForRailEnv(max_depth=params.tree_depth, predictor=predictor)
        
    # Input dims
    number_of_node_feats = obs_object.observation_dim #Features per node
    number_of_nodes = 0
    for count in range(params.tree_depth + 1):
        number_of_nodes += np.power(4,count) #Each node can be expanded to 4 nodes

    input_dims = number_of_nodes * number_of_node_feats

    # Pick agent in eval mode
    if params.alg == "DQN":
        agent = DQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=True)
    elif params.alg == "DDQN":
        agent = DDQNsoftPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=True)
    elif params.alg == "DDDQN":
        agent = DDDQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=True)
    elif params.alg == "DRQN":
        agent = DRQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=True)
    elif params.alg == "PPO":
        agent = PPOPolicy(input_dims=input_dims, n_actions=env_params.n_actions, n_agents=env_params.n_agents, params=params, evaluation_mode=True)
    elif params.alg in ppo_policies:
        agent = PPORPolicy(input_dims=input_dims, n_actions=env_params.n_actions, n_agents=env_params.n_agents, params=params, evaluation_mode=True)
    else:
        sys.exit("Wrong algorithm input")

    levels = ['easy', 'medium', 'hard']
    level = levels[params.level-1]

    means_scores = []
    means_completions = []
    scores_dict = dict()
    for training_seed in range(1,11):
        # Re-start environment
        env = get_env(env_params, obs_object, params.seed)
        env.reset(True, True)

        # Setup renderer
        env_renderer = RenderTool(env, gl='PGL')

        # Set checkpoint
        checkpoint = os.path.join('checkpoints', 'official', level, params.alg, str(training_seed), 'net.pth')
        
        # Load model
        if os.path.isfile(checkpoint):
            agent.load(checkpoint)
        else:
            print("Checkpoint not found, using untrained policy! (path: {})".format(checkpoint))
        
        # Begin evaluation
        scores, completions = evaluate_agent(env, agent, params, env_params, training_seed)
        
        # Store results
        scores_dict.update({training_seed: scores})
        means_scores.append(np.mean(scores))
        means_completions.append(np.mean(completions))

    avg_returns = np.average(means_scores)
    std_returns = np.std(means_scores)
    avg_complete = np.average(means_completions)
    std_complete = np.std(means_completions)
    scores_dict.update({'average': avg_returns})
    scores_dict.update({'Standard Dev': std_returns})
    dframe = pd.DataFrame(scores_dict)
    
    results_directory = os.path.join('official_eval', level)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    results_directory = os.path.join(results_directory, params.alg+'.csv')
    dframe.to_csv(results_directory)

def evaluate_agent(env, agent, params, env_params, training_seed):
    max_steps = env._max_episode_steps
    scores = []
    completions = []
    l_steps = []
    # Evaluate current model state for ten episodes 
    for episode in range(params.n_episodes + 1):
        update_values = [False]*env_params.n_agents
        obs = [None]*env_params.n_agents
        actions = dict()
        score = 0
        env_obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        
        for h in env.get_agent_handles():
            obs[h] = normalize_observation(env_obs[h], params.tree_depth, params.tree_radius)
            if params.alg == "DRQN":
                agent.history[h].clear()
                obs[h] = agent.stack_states(obs[h], h)                    

        for t_step in range(max_steps - 1):
            

            if check_if_all_blocked(env):
                #If all agents blocked end the episode
                steps_left = max_steps - t_step - 1
                blocked_agents = sum(not dones[h] for h in env.get_agent_handles())
                score -= steps_left*blocked_agents
                break
            
            for h in env.get_agent_handles():
                if info['action_required'][h]:                    
                    update_values[h] = True
                    action = agent.act(obs[h], eval=True)
                else:
                    update_values[h] = False
                    action = 0
                actions.update({h: action})

            # Perform actions on environment
            env_obs, rewards, dones, info = env.step(actions)

            for h in env.get_agent_handles():
                if env_obs[h]:
                    obs[h] = normalize_observation(env_obs[h], params.tree_depth, params.tree_radius)
                
                if params.alg == "DRQN":
                    agent.history[h].clear()
                    obs[h] = agent.stack_states(obs[h], h)                    

                score += rewards[h]
            
            last_step = t_step
            if dones['__all__']:
                break
        
        if (episode+1) == params.n_episodes:
            end = '\n'
        else:
            end = ' '

        print('\rEvaluating {} on Level: {}\t Training seed: {}\t Evaluation episode: {}\t'.format(
            params.alg,
            str(params.level),
            training_seed,
            episode+1
        ), end=end)

        normalized_score = score / (max_steps * env_params.n_agents)
        scores.append(normalized_score)
        tasks_finished = sum([int(dones[idx]) for idx in env.get_agent_handles()])
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)
        l_steps.append(last_step)

    return scores, completions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alg", help="Algorithm", default="PPOR", type=str)
    parser.add_argument("--n_episodes", help="number of episodes", default=10, type=int)
    parser.add_argument("--level", help="Level difficulty: [1, 2, 3]", default=1, type=int)
    parser.add_argument("--tree_depth", help="Max observation depth", default=2, type=int)
    parser.add_argument("--tree_radius", help="Max observation radius", default=10, type=int)
    parser.add_argument("--predictor", help="Use predictor", default=True, type=bool)
    parser.add_argument("--hidden_layer", help="Hidden layers dimension", default=125, type=int)
    parser.add_argument("--lrate", help="Learning rate", default=0.0009, type=float)
    parser.add_argument("--gamma", help="Importance of later rewards", default=0.99, type=float)
    parser.add_argument("--epsilon", help="Initial epsilon", default=1.00, type=float)
    parser.add_argument("--eps_min", help="minimum epsilon", default=0.05, type=float)
    parser.add_argument("--eps_decay", help="epsilon decay", default=0.997, type=float)
    parser.add_argument("--buffer_size", help="Replay Buffer max length", default=80000, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=160, type=int)
    parser.add_argument("--learn_freq", help="Learn frequency", default=8, type=int)
    parser.add_argument("--update_target_freq", help="Target network update frequency", default=8, type=int)
    parser.add_argument("--n_layers", help="Number of LSTM layers", default=3, type=int)
    parser.add_argument("--history_length", help="Length of the state history", default=15, type=int)
    parser.add_argument("--priority", help="Use Priority Buffer", default=False, type=bool)
    parser.add_argument("--render", help="Render the environment", default=False, type=bool)
    parser.add_argument("--use_gpu", help="Use CUDA", default=True, type=bool)
    parser.add_argument("--seed", help="Set the seed", default=12, type=int)
    params = parser.parse_args()
    main(params)