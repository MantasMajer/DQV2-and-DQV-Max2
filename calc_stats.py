import sys
import numpy as np

FILE_PATH = "results/q_weight_exps/"

if __name__ == "__main__":
    if len(sys.argv):
        #algorithm_names = ["DQV-Max2", "DQV2", "DQV", "DQV-Max", "DQN"]
        algorithm_names = ["DQV", "0.1", "0.2", "0.3", "0.4", "0.5"]
        #algorithm_names = ["DQV-Max", "0.1", "0.2", "0.3", "0.4", "0.5"]
        train_results = np.load(FILE_PATH + "train_results_" + sys.argv[1] + ".npy")
        exploit_results = np.load(FILE_PATH + "exploit_results_" + sys.argv[1] + ".npy")

        total_train_rewards = np.sum(train_results, axis=2)
        total_exploit_rewards = np.sum(exploit_results, axis=2)
        for i in range(len(algorithm_names)):
            mean_train_reward = np.mean(total_train_rewards, axis=1)
            mean_exploit_reward = np.mean(total_exploit_rewards, axis=1)
            std_train_reward = np.std(total_train_rewards, axis=1)
            std_exploit_reward = np.std(total_exploit_rewards, axis=1)
            print(algorithm_names[i])
            print("training: mean = " + str(int(mean_train_reward[i])) + " std: " + str(int(std_train_reward[i]))) 
            print("exploitation: mean = " + str(int(mean_exploit_reward[i])) + " std: " + str(int(std_exploit_reward[i])))
            print("")