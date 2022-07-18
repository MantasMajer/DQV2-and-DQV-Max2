# Install packages
from tabnanny import verbose
import gym
from PIL import Image
from IPython.display import clear_output
import numpy as np
from Conv_DQN import Conv_DQN
from DQN import DQN
from DQV import DQV
from DQV_Max import DQV_Max
from DQV2 import DQV2
from DQV_Max2 import DQV_Max2


EXPERIMENTS = 10
EXPLOITATION_PERIOD = 10
# directory where to save results
SAVE_PATH = "results/q_weight_exps/"

algorithms = [DQV_Max2, DQV2, DQV, DQV_Max, DQN]
#algorithms = [DQV2] * 6
#algorithms = [DQV_Max2] * 6

algorithm_names = ["DQV-Max2", "DQV2", "DQV", "DQV-Max", "DQN"]
#algorithm_names = ["DQV", "0.1", "0.2", "0.3", "0.4", "0.5"]
#algorithm_names = ["DQV-Max", "0.1", "0.2", "0.3", "0.4", "0.5"]

envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
env_names = ["CartPole_2", "Acrobot_2", "MountainCar"]

for k in range(len(envs)):

    env_name = env_names[k]
    print(env_name)
    env = gym.envs.make(envs[k])

    n_state = env.observation_space.shape
    print(n_state)
    if len(n_state) == 1:
        n_state = n_state[0]
    n_action = env.action_space.n
    episodes = 500
    n_hidden = 50 # number of neurons in the hidden layer in the MLP
    lr = 0.001


    exploit_results_length = int((episodes + 9) / EXPLOITATION_PERIOD)

    # for saving the results
    training_results_raw = np.zeros((len(algorithms), EXPERIMENTS, episodes))
    exploit_results_raw = np.zeros((len(algorithms), EXPERIMENTS, exploit_results_length))

    # test each algorithm
    for i, algorithm in enumerate(algorithms):
        print("Now dealing with " + algorithm_names[i])
        # run a number of experiments for each algorithm
        for j in range(EXPERIMENTS):
            model = algorithm(n_state, n_action, n_hidden, lr, q_weight=i*0.1)
            train_results, exploit_results = model.train(env, episodes, gamma=.99, verbose=False,
                                            epsilon=0.5, replay_size=16, buffer_size=65536, 
                                            exploitation_period=EXPLOITATION_PERIOD, exploitation_runs=3)
            training_results_raw[i][j] = train_results
            exploit_results_raw[i][j] = exploit_results

    # save the results
    np.save(SAVE_PATH + "train_results_" + env_name, training_results_raw)
    np.save(SAVE_PATH + "exploit_results_" + env_name, exploit_results_raw)
