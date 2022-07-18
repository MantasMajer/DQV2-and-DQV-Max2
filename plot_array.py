import sys
import numpy as np
import matplotlib.pyplot as plt



def plot_res(results_training, results_exploitation, algorithm_names, title=''):   
    
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    for i in range(len(algorithm_names)):
        ax[0].plot(results_training[i], label=algorithm_names[i])
        ax[1].plot(results_exploitation[i], label=algorithm_names[i])
    ax[0].set_xlabel('Episodes')
    ax[1].set_xlabel('Increments of 10 episodes')
    ax[0].set_ylabel('Reward')
    ax[1].set_ylabel('Reward')
    ax[0].legend()
    ax[1].legend()
    
    plt.show()

FILE_PATH = "results/q_weight_exps/"

def main():
    if len(sys.argv) == 2:
        """ uncomment the algorithm names needed """
        #algorithm_names = ["DQV-Max2", "DQV2", "DQV", "DQV-Max", "DQN"]
        #algorithm_names = ["DQV", "0.1", "0.2", "0.3", "0.4", "0.5"]
        algorithm_names = ["DQV-Max", "0.1", "0.2", "0.3", "0.4", "0.5"]
        train_results = np.load(FILE_PATH + "train_results_" + sys.argv[1] + ".npy")
        exploit_results = np.load(FILE_PATH + "exploit_results_" + sys.argv[1] + ".npy")
        mean_training = list(np.mean(train_results, axis=1))
        mean_exploitation = list(np.mean(exploit_results, axis=1))
        plot_res(mean_training, mean_exploitation, algorithm_names)



if __name__ == "__main__":
    main()