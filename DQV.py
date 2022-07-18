from collections import deque
import random
from DQN import DQN
from Conv_DQN import Conv_DQN
import numpy as np
import torch
import copy

class DQV():

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05, conv=False):
        self.conv = conv
        if conv:
            self.Q_net = Conv_DQN(state_dim, action_dim, lr=lr)
            self.V_net = Conv_DQN(state_dim, 1, lr=lr)
        else:
            self.Q_net = DQN(state_dim, action_dim, hidden_dim=hidden_dim, lr=lr)
            self.V_net = DQN(state_dim, 1, hidden_dim=hidden_dim, lr=lr)

    def update_V(self, state, y):
        self.V_net.update(state, y)
    
    def update_Q(self, state, y):
        self.Q_net.update(state, y)

    def predict_Q(self, state):
        return self.Q_net.predict(state)

    def predict_V(self, state):
        return self.V_net.predict(state)


    def exploit(self, env):
        state = env.reset()
        total = 0
        done = False
        while not done:
            q_values = self.predict_Q(state)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total += reward
            state = next_state
        return total

    def train(self, env, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99, replay_size=16, 
               buffer_size=200, verbose=True, exploitation_period=10, exploitation_runs=10):
        final = np.zeros(episodes)
        no_train = np.zeros(int((episodes + 9) / exploitation_period))
        memory = deque(maxlen=buffer_size)
        best_model = self
        best_performance = 0

        for episode in range(episodes):

            # every once in a while, run the best model with a greedy strategy 
            if episode % exploitation_period == exploitation_period - 1:
                exp_reward = 0
                for _ in range(exploitation_runs):
                    exp_reward += best_model.exploit(env)
                exp_reward /= exploitation_runs
                # save the average reward obtained
                no_train[int(episode / exploitation_period)] = exp_reward
                best_performance = exp_reward
                print("average pure exploitation performance: " + str(exp_reward))
            
            # Reset state
            state = env.reset()
            done = False
            total = 0
            
            while not done:
                # select the action based on the epsilon-greedy exploration strategy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = self.predict_Q(state)
                    action = torch.argmax(q_values).item()
                
                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)
                
                # Update total and memory
                total += reward
                memory.append((state, action, next_state, reward, done))

                if done:
                    break

                # Update network weights using replay memory
                self.replay_fast(memory, replay_size, gamma)

                state = next_state
            
            # Update epsilon
            epsilon = max(epsilon * eps_decay, 0.01)
            final[episode] = total

            # if the current model performed better than the best one, test the model again
            if total > best_performance:
                total = (total + self.exploit(env)) / 2
                if total > best_performance:
                    # update the best model estimate
                    best_model = copy.deepcopy(self)
                    best_performance = total
            
            if verbose:
                print("episode: {}, total reward: {}".format(episode + 1, total))

        print("Training finished")
        return final, no_train


    def replay_fast(self, memory, size, gamma=0.9):
        
        if len(memory) >= size:
            batch = random.sample(memory, size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = np.array(batch_t[0])
            actions = np.array(batch_t[1])
            next_states = np.array(batch_t[2])
            rewards = np.array(batch_t[3])
            is_dones = np.array(batch_t[4])

            # if we are dealing with image data, the states need to be transposed
            if self.conv:
                states = np.transpose(states, (0, 3, 1, 2))
                next_states = np.transpose(next_states, (0, 3, 1, 2))
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            # because terminal states will not have future rewards
            is_dones_indices = torch.where(is_dones_tensor == True)[0].tolist()

            # calculate V targets
            v_values_next = self.V_net.model(next_states)
            v_values_next[:, 0] = torch.mul(v_values_next, gamma)[:, 0] + rewards
            v_values_next[is_dones_indices, 0] = rewards[is_dones_indices]

            # update V values
            self.update_V(states, v_values_next)

            v_values_next = self.V_net.model(next_states)
            all_q_values = self.Q_net.model(states)

            # Calculste TD targets for the Q values
            all_q_values[range(len(all_q_values)), actions] = rewards + torch.mul(v_values_next, gamma)[:, 0]
            all_q_values[is_dones_indices, actions_tensor[is_dones_indices].tolist()] = rewards[is_dones_indices]
                   
            # update the Q values
            self.update_Q(states, all_q_values)

