import torch as T
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from utils.misc import get_level, log_results, get_checkpoint_path
from utils.deadlock_check import check_for_deadlock, check_if_all_blocked, get_agent_positions
from utils.observation_utils import normalize_observation
from torch.utils.tensorboard import SummaryWriter
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from reinforcement_learning.policy_optimization import PPOPolicy, PPORPolicy
from reinforcement_learning.value_optimization import DDDQNPolicy, DDQNsoftPolicy, DRQNPolicy, DQNPolicy
from argparse import ArgumentParser, Namespace
from collections import deque
import numpy as np
import datetime
import random
import time
import sys
import os

# Algorithm definitions. If different algorithms are to be tested, update the lists.
value_policies = ['DQN', 'DDQN', 'DDDQN', 'DRQN']
ppo_policies = ['PPO', 'PPOR' ,'SIL']
level = ['easy', 'medium', 'hard']

def main(params, seed):
    if params.log_results:
        writer = log_results(params, level)
    
    # Set environment parameters 
    env_params =  Namespace(**get_level(difficulty=params.level, seed=seed))

    #Set the seed
    random.seed(seed)
    np.random.seed(seed)

    # My predictor
    predictor = ShortestPathPredictorForRailEnv(params.tree_depth)

    # My observation 
    obs_object = TreeObsForRailEnv(max_depth=params.tree_depth,predictor=predictor)

    # Generate env object
    env = get_env(env_params, obs_object, seed)
    env.reset(True, True)

    #Renderer 
    env_renderer = RenderTool(env, gl="PGL")
    
    # Calculate state size
    number_of_node_feats = obs_object.observation_dim #Features per node
    number_of_nodes = 0
    for count in range(params.tree_depth + 1):
        number_of_nodes += np.power(4,count) #Each node can be expanded to 4 nodes

    # Set network parameters
    input_dims = number_of_nodes * number_of_node_feats
    
    # Initialize agent
    if params.alg == "DQN":
        agent = DQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=False)
    elif params.alg == "DDQN":
        agent = DDQNsoftPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=False)
    elif params.alg == "DDDQN":
        agent = DDDQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=False)
    elif params.alg == "DRQN":
        agent = DRQNPolicy(input_dims=input_dims, n_actions=env_params.n_actions, params=params, evaluation_mode=False)
    elif params.alg == "PPO" or params.alg == "SIL":
        agent = PPOPolicy(input_dims=input_dims, n_actions=env_params.n_actions, n_agents=env_params.n_agents, params=params, evaluation_mode=False)
    elif params.alg in ppo_policies:
        agent = PPORPolicy(input_dims=input_dims, n_actions=env_params.n_actions, n_agents=env_params.n_agents, params=params, evaluation_mode=False)
    else:
        sys.exit("Wrong algorithm input")

    checkpoint_path = get_checkpoint_path(params, level)

    # Load a trained model stored into checkpoint_path
    if params.load_model:
        agent.load(checkpoint_path+'/net.pth')
    
    # Experiment parameters
    # Official formula of max_steps by FlatLand team
    max_steps = int(4 * 2 * (env.height + env.width + (env_params.n_agents / env_params.max_cities)))
    actions = dict()
    obs = [None]*env_params.n_agents
    next_obs = [None]*env_params.n_agents  
    update_values = [False]*env_params.n_agents
    mask = [False]*env_params.n_agents
    logprob = [0]*env_params.n_agents
    v_hidden = [None]*env_params.n_agents
    a_hidden = [None]*env_params.n_agents
    
    # Statistics variables
    scores = []
    scores_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    completion = []
    step_times = []
    action_count = [0] * 5
    timestep = 0
    loss = 0
    frame_step = 0

    # Build global means and stds for normalized Returns
    if params.alg in ppo_policies:
        agent.build_means_and_stds(max_steps)

    # Start Training
    for episode in range(params.n_episodes + 1):
        score = 0
        episode_step_times = []

        # Reset environment
        env_obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        
        if params.render:
            env_renderer.set_new_rail()
        
        # Normalize observations
        for h in env.get_agent_handles():
            obs[h] = normalize_observation(env_obs[h], params.tree_depth, params.tree_radius)
            if params.alg == "DRQN":
                agent.history[h].clear()
                obs[h] = agent.stack_states(obs[h], h)
                v_hidden[h] = (T.zeros(params.n_layers, 1, params.hidden_layer).to('cuda:0'),
                        T.zeros(params.n_layers, 1, params.hidden_layer).to('cuda:0'))
                a_hidden[h] = (T.zeros(params.n_layers, 1, params.hidden_layer).to('cuda:0'),
                        T.zeros(params.n_layers, 1, params.hidden_layer).to('cuda:0'))
                
        for t_step in range(max_steps-1):
                        
            timestep+=1
            episode_length = t_step
            #If all agents blocked end the episode
            if check_if_all_blocked(env) and params.alg in value_policies:
                steps_left = max_steps - t_step - 1
                blocked_agents = sum(not dones[h] for h in env.get_agent_handles())
                score -= steps_left*blocked_agents
                break
            
            for h in env.get_agent_handles():
                if info['action_required'][h]:                    
                    update_values[h] = True
                    if params.alg == "DRQN":
                        action, v_hidden[h], a_hidden[h] = agent.act(obs[h], v_hidden[h], a_hidden[h])
                    elif params.alg in ppo_policies: 
                        action, logprob[h] = agent.act(obs[h])
                    else:    
                        action = agent.act(obs[h])
                    action_count[action] += 1
                else:
                    logprob[h] = 0
                    update_values[h] = False
                    action = 0
                actions.update({h: action})

            # Perform actions on environment
            prev_distance = get_distances(env)
            env_obs, rewards, dones, info = env.step(actions)
            
            tmp_rewards = rewards.copy()

            timer_step_start = time.time()
            for h in env.get_agent_handles():
                
                ######### Experimenting with different rewards
                #rewards[h] = build_rewards(h, env, dones, prev_distance[h])
                if env_obs[h]:
                    next_obs[h] = normalize_observation(env_obs[h],
                                params.tree_depth, params.tree_radius)

                    if params.alg == "DRQN":
                        next_obs[h] = agent.stack_states(next_obs[h], h)
                                
                if (update_values[h] or dones['__all__']) and params.alg in value_policies:
                    agent.step(obs[h], actions[h], rewards[h], next_obs[h], dones[h], h)                    
                    obs[h] = next_obs[h].copy()                
                score += tmp_rewards[h]
                
            #if (sum(update_values) or dones['__all__']) and params.alg in ppo_policies:
            if params.alg in ppo_policies:
                agent.step(obs.copy(), actions.copy(), rewards.copy(), next_obs.copy(), dones.copy(), update_values.copy(), logprob.copy())
                obs = [nob.copy() for nob in next_obs]    
            
            timer_step_end = time.time()
            if params.render:
                env_renderer.render_env(show=True, frames=False, show_observations=False, 
                                        show_predictions=False)

            if dones['__all__']:
                break
    
            episode_step_times.append(timer_step_end-timer_step_start)
            
            # Render / GIF zone
            if params.make_video:
                env_renderer.gl.save_image("Images/"+level[params.level - 1]+"/"+"untrained"+"/flatland_frame_{:04d}.bmp".format(frame_step))
                frame_step+=1
        
        if params.alg in value_policies:
            agent.update_epsilon()  # Slowly decay epsilon 
        else:
            agent.end_episode()
            loss = agent.learn()
            if params.alg == "SIL":
                agent.learn_sil()
        

        # Statistics zone
        action_probs = action_count / max(1, np.sum(action_count))
        tasks_finished = np.sum([int(dones[idx]) for idx in env.get_agent_handles()])
        normalized_completion = tasks_finished / max(1, env.get_num_agents())
        completion_window.append(normalized_completion)
        normalized_score = score / (max_steps * env.get_num_agents())
        scores_window.append(normalized_score)
        completion.append((np.mean(completion_window)))
        scores.append(np.mean(scores_window))
        step_times.append(np.mean(episode_step_times))

        if episode % 100 == 0:
            action_count = [0] * 5
            end = "\n"
        else:
            end = " "
        
        print('\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f}\tAction Probs: {}'.format(
            env.get_num_agents(),
            env_params.width, env_params.height,
            episode,
            np.mean(scores_window),
            100 * np.mean(completion_window),
            0.0,
            #agent.epsilon,
            format_action_prob(action_probs)
        ), end=end)
        
        if log_results:
            writer.add_scalar('training/nomralized_scores', normalized_score, episode+1)
            writer.add_scalar('training/mean_scores', np.mean(scores_window), episode+1)
            writer.add_scalar('training/normalized_completions', normalized_completion, episode+1)
            writer.add_scalar('training/mean_completions', np.mean(completion_window)*100, episode+1)
            writer.add_scalar('training/episode_mean_step_times', np.mean(episode_step_times), episode+1)
            writer.add_scalar('training/loss', loss, episode+1)

            
            writer.add_scalar('Metrics/NormalizedReturn', normalized_score, (episode+1))
            writer.add_scalar('Metrics/AverageReturn', np.mean(scores_window), (episode+1))
            writer.add_scalar('Metrics/NumberOfEpisodes', (episode+1), (episode+1))

            writer.add_scalar('Metrics_vs_EnvironmentSteps/AverageEpisodeLength', episode_length, timestep)
            writer.add_scalar('Metrics_vs_EnvironmentSteps/NormalizedReturn', normalized_score, timestep)
            writer.add_scalar('Metrics_vs_EnvironmentSteps/AverageReturn', np.mean(scores_window), timestep)
            writer.add_scalar('Metrics_vs_EnvironmentSteps/NumberOfEpisodes', (episode+1), timestep)

            writer.add_scalar('Metrics_vs_NumberOfEpisodes/AverageEpisodeLength', episode_length, (episode+1))
            writer.add_scalar('Metrics_vs_NumberOfEpisodes/NormalizedReturn', normalized_score, (episode+1))
            writer.add_scalar('Metrics_vs_EnvironmentSteps/AverageReturn', np.mean(scores_window), timestep)
            writer.add_scalar('Metrics_vs_NumberOfEpisodes/EnvironmentSteps', timestep, (episode+1))

    if log_results:
        writer.close()
    # Save trained agent
    if params.save_model:
        agent.save(checkpoint_path+'/net.pth')
        

def get_env(params, obs_object, seed):
    env = RailEnv(
        width=params.width,
        height=params.height,
        rail_generator=sparse_rail_generator(
            max_num_cities=params.max_cities, 
            grid_mode=False, 
            max_rails_between_cities=params.max_rails_between_cities,
            max_rails_in_city=params.max_rails_in_city,
            seed=params.seed),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=params.n_agents,
        obs_builder_object=obs_object,
        malfunction_generator_and_process_data=None,
        malfunction_generator=None,
        remove_agents_at_target=True,
        random_seed=seed,
        record_steps=False,
        close_following=True
    )
    return env

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer

def make_directory(name):
    path = os.getcwd()
    date = datetime.datetime.now().strftime("%x").replace('/', '_')
    time = datetime.datetime.now().strftime("%X")
    folder = name+'_'+date+'_'+time
    if not os.path.isdir('stats'):
        os.mkdir(path+'/stats')
    try:
        os.mkdir(path+'/stats/'+folder)
    except OSError:
        print("Stats storing failed")
    else:
        pass
    
    return 'stats/'+folder


def get_distances(env):
    distance_map = env.distance_map.get()
    agent_position = []
    agent_direction = []
    distance_maps = []
    for h in env.get_agent_handles():
        agent_position.append(env.agents[h].position if env.agents[h].position is not None else env.agents[h].initial_position)
        agent_direction.append(env.agents[h].direction if env.agents[h].direction is not None else env.agents[h].initial_direction)
        distance_maps.append(distance_map[(h, *agent_position[h], agent_direction[h])])
    return distance_maps

def build_rewards(h, env, dones, prev_distance):
    deadlock = check_for_deadlock(h, env, get_agent_positions(env), env.agents[h].position, env.agents[h].direction)
    distance_map = env.distance_map.get()
    agent_position = env.agents[h].position if env.agents[h].position is not None else env.agents[h].initial_position
    agent_direction = env.agents[h].direction if env.agents[h].direction is not None else env.agents[h].initial_direction
    distance = distance_map[(h, *agent_position, agent_direction)]
    dd = distance - prev_distance
    return (0.01*1)-5*deadlock + 10*dones[h]

if __name__ == "__main__":
    parser = ArgumentParser()
    ################## GENERAL ##################
    parser.add_argument("--alg", help="Algorithm", default="SIL", type=str)
    parser.add_argument("--n_episodes", help="number of episodes", default=1500, type=int)
    parser.add_argument("--level", help="Level difficulty: [1, 2, 3]", default=2, type=int)
    parser.add_argument("--tree_depth", help="Max observation depth", default=2, type=int)
    parser.add_argument("--tree_radius", help="Max observation radius", default=10, type=int)
    parser.add_argument("--predictor", help="Use predictor", default=True, type=bool)
    parser.add_argument("--hidden_layer", help="Hidden layers dimension", default=125, type=int)
    parser.add_argument("--lrate", help="Learning rate", default=0.0005, type=float) # 0.0005 if ppo
    parser.add_argument("--gamma", help="Importance of later rewards", default=0.99, type=float)
    parser.add_argument("--batch_size", help="Batch size", default=300, type=int)
    parser.add_argument("--buffer_size", help="Replay Buffer max length", default=40000, type=int) #50000 if ppo_sil
    parser.add_argument("--priority", help="Activate Prioritized Replay", default=False, type=bool)
    ################## VALUE ##################
    parser.add_argument("--epsilon", help="Initial epsilon", default=1.00, type=float)
    parser.add_argument("--eps_min", help="minimum epsilon", default=0.05, type=float)
    parser.add_argument("--eps_decay", help="epsilon decay", default=0.997, type=float)
    parser.add_argument("--learn_freq", help="Learn frequency", default=50, type=int)
    parser.add_argument("--update_target_freq", help="Target network update frequency", default=8, type=int)
    parser.add_argument("--n_layers", help="Number of LSTM layers", default=1, type=int)
    parser.add_argument("--history_length", help="Length of the state history", default=1, type=int)
    parser.add_argument("--alpha", help="alpha parameter for Prioritized Replay", default=0.7, type=float)
    parser.add_argument("--alpha_decay", help="alpha parameter for Prioritized Replay", default=0.999, type=float)
    parser.add_argument("--beta", help="alpha parameter for Prioritized Replay", default=0.4, type=float)
    parser.add_argument("--beta_rise", help="alpha parameter for Prioritized Replay", default=1.006, type=float)
    ################## POLICY ##################
    parser.add_argument("--tau", help="tau for GAE", default=0.95, type=float)
    parser.add_argument("--clip", help="bound of the policy ratio", default=0.3, type=float)
    parser.add_argument("--c1", help="Weight of the Value error in the Loss function", default=0.5, type=float)
    parser.add_argument("--c2", help="Weight of the entropy in the Loss function", default=0.01, type=float)
    parser.add_argument("--epochs", help="PPO learn epochs", default=10, type=int)
    parser.add_argument("--update_policy", help="Every when to update policy", default=80, type=int)
    parser.add_argument("--sil_epochs", help="Self Imitation epochs", default=13, type=int)
    ################## TECHNICAL ##################
    parser.add_argument("--load_model", help="Load a trained model", default=False, type=bool)
    parser.add_argument("--save_model", help="Save a trained model", default=False, type=bool)
    parser.add_argument("--evaluation_n_episodes", help="number of evaluation episodes", default=10, type=int)
    parser.add_argument("--render", help="Render the environment", default=False, type=bool)
    parser.add_argument("--use_gpu", help="Use CUDA", default=True, type=bool)
    parser.add_argument("--seed", help="Set the seed", default=1, type=int)
    parser.add_argument("--comment", help="Add a comment to Tensorboard run", default="", type=str)
    parser.add_argument("--make_video", help="Make a video of the run", default=False, type=bool)
    parser.add_argument("--log_results", help="Log results to TensorBoard", default=False, type=bool)
    params = parser.parse_args()
    main(params, params.seed)