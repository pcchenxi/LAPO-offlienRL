#!/usr/bin/env python3
"""
Based on https://github.com/sfujim/BCQ
"""
import argparse, os, torch
import numpy as np
from algos import utils
from logger import logger, setup_logger
import matplotlib.pyplot as plt
from matplotlib import cm
import algos.algos_vae as algos
import gym, d4rl

def eval_policy(policy, env, eval_episodes=10, plot=False):
    avg_reward = 0.
    plt.clf()
    start_states = []
    color_list = cm.rainbow(np.linspace(0, 1, eval_episodes+2))

    for i in range(eval_episodes):
        state, done = env.reset(), False
        states_list = []
        start_states.append(state)
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            states_list.append(state)
        states_list = np.array(states_list)

        if plot:
            plt.scatter(states_list[:,0], states_list[:,1], color = color_list[i], alpha=0.1)
            plt.scatter(8, 10, color = 'white', alpha=0.1)
            plt.scatter(2, 0, color = 'white', alpha=0.1)
    if plot:
        start_states = np.array(start_states)
        plt.scatter(start_states[:,0], start_states[:,1], color='red')
        plt.savefig('./eval_fig') 

    avg_reward /= eval_episodes
    normalized_score = env.get_normalized_score(avg_reward)

    info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
    print ("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
    print ("---------------------------------------")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--ExpID", default=0000, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=0, type=float)          # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=5e5, type=int)           # How often it saves the model
    parser.add_argument("--env_name", default="maze2d-large-v1")     # OpenAI gym environment name
    parser.add_argument("--seed", default=123, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)      # Max time steps to run environment for
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--vae_lr', default=2e-4, type=float)	        # action policy (VAE) learning rate
    parser.add_argument('--actor_lr', default=2e-4, type=float)	        # latent policy learning rate
    parser.add_argument('--critic_lr', default=2e-4, type=float)	    # critic learning rate
    parser.add_argument('--tau', default=0.005, type=float)	            # delayed learning rate
    parser.add_argument('--discount', default=0.99, type=float)	        # discount factor

    parser.add_argument('--expectile', default=0.9, type=float)	        # expectile to compute weight for samples
    parser.add_argument('--kl_beta', default=1, type=float)	            # weight for kl loss to train CVAE
    parser.add_argument('--max_latent_action', default=2.0, type=float)	# maximum value for the latent policy
    parser.add_argument('--doubleq_min', default=1, type=float)         # weight for the minimum Q value
    parser.add_argument('--no_noise', action='store_true')              # adding noise to the latent policy or not

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cpu', type=str)

    args = parser.parse_args()

    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}/{args.env_name}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if os.path.exists(os.path.join(folder_name, 'progress.csv')):
        print('exp file already exist')
        raise AssertionError

    variant = vars(args)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

    # Setup Environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(action_dim)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Dataset
    dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    if 'antmaze' in args.env_name:
        dataset['rewards'] = (dataset['rewards']*100)
        min_v = 0
        max_v = 1 * 100
    else:
        dataset['rewards'] = dataset['rewards']/dataset['rewards'].max()
        min_v = dataset['rewards'].min()/(1-args.discount)
        max_v = dataset['rewards'].max()/(1-args.discount)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.device, max_size=len(dataset['rewards']))
    replay_buffer.load(dataset)

    latent_dim = action_dim*2
    policy = algos.Latent(state_dim, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer=replay_buffer, 
                        device=args.device, discount=args.discount, tau=args.tau, 
                        vae_lr=args.vae_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
                        max_latent_action=args.max_latent_action, expectile=args.expectile, kl_beta=args.kl_beta, 
                        no_noise=args.no_noise, doubleq_min=args.doubleq_min)

    if args.load_model != 0:
        policy.load('model_' + str(args.load_model), folder_name)
        training_iters = int(args.load_model)
    else:
        training_iters = 0

    while training_iters < args.max_timesteps:
        # Train
        pol_vals = policy.train(iterations=int(args.eval_freq)
                                , batch_size=args.batch_size)
        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))

        # Save Model
        if training_iters % args.save_freq == 0 and args.save_model:
            policy.save('model_' + str(training_iters), folder_name)

        # Eval
        info = eval_policy(policy, env, plot=args.plot)
        for k, v in info.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()

    policy.save('model_' + str(training_iters), folder_name)
