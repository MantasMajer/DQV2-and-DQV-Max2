from collections import deque
import torch
from torch.autograd import Variable
import random
import numpy as np
import copy

def calc_dim(x, kernel_size, stride):
    return int((x -(kernel_size - 1) - 1) / stride) + 1 


class Conv_DQN(torch.nn.Module):
    ''' Deep Q Neural Network class. '''
    def __init__(self, input_shape, action_dim, lr=0.05, conv=True):
        super().__init__()
        h_out_conv = calc_dim(calc_dim(calc_dim(input_shape[0], 8, 4), 4, 2), 3, 1)
        w_out_conv = calc_dim(calc_dim(calc_dim(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(input_shape[2], 32, (8, 8), stride=4),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 64, (4, 4), stride=2),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(64, 64, (3, 3)),
                        torch.nn.ReLU(),
                        torch.nn.Flatten(),
                        torch.nn.Linear(w_out_conv * h_out_conv * 64, 512),
                        torch.nn.Linear(512, action_dim)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)



    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(state)
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        state_t = np.transpose(state, (2, 0, 1))
        with torch.no_grad():
            return self.model(torch.Tensor(state_t[None, ...]))

    def exploit(self, env):
        state = env.reset()
        total = 0
        done = False
        while not done:
            q_values = self.predict(state)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total += reward
            state = next_state
        return total


    def train(self, env, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99, replay_size=16,  
               verbose=True, buffer_size=50, exploitation_period=10, exploitation_runs=10):
        """Deep Q Learning algorithm using the DQN. """
        final = np.zeros(episodes)
        no_train = np.zeros(int((episodes + 9) / exploitation_period))
        memory = deque(maxlen=buffer_size)
        best_model = self
        best_performance = 0

        for episode in range(episodes):

            if episode % exploitation_period == exploitation_period - 1:
                exp_reward = 0
                for _ in range(exploitation_runs):
                    exp_reward += best_model.exploit(env)
                no_train[int(episode / exploitation_period)] = exp_reward / exploitation_runs
                print("average pure exploitation performance: " + str(exp_reward / exploitation_runs))
            
            # Reset state
            state = env.reset()
            done = False
            total = 0
            
            while not done:
                # Implement greedy search policy to explore the state space
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = self.predict(state)
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

            if total > best_performance:
                total = (total + self.exploit(env)) / 2
                if total > best_performance:
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
            states = np.transpose(np.array(batch_t[0]), (0, 3, 1, 2))
            actions = np.array(batch_t[1])
            next_states = np.transpose(np.array(batch_t[2]), (0, 3, 1, 2))
            rewards = np.array(batch_t[3])
            is_dones = np.array(batch_t[4])
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0].tolist()
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            #Update q values
            all_q_values[range(len(all_q_values)),actions] = rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices, actions_tensor[is_dones_indices].tolist()] = rewards[is_dones_indices]
        
            self.update(states, all_q_values)