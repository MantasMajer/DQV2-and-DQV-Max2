from DQV2 import DQV2
from DQN import DQN
import torch
import numpy as np
import random
import time

class DQV_Max2(DQV2):

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05, q_weight=0.5, conv=False):
        super().__init__(state_dim, action_dim, hidden_dim, lr, q_weight=q_weight, conv=conv)


    def replay_fast(self, memory, size, gamma=0.9):
        
        if len(memory) >= size:
            batch = random.sample(memory, size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = np.array(batch_t[0])
            actions = np.array(batch_t[1])
            next_states = np.array(batch_t[2])
            rewards = np.array(batch_t[3])
            is_dones = np.array(batch_t[4])

            if self.conv:
                states = np.transpose(states, (0, 3, 1, 2))
                next_states = np.transpose(next_states, (0, 3, 1, 2))
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor == True)[0].tolist()

            # calculate v targets
            all_q_values_next = self.Q_net.model(next_states)
            all_q_values = self.Q_net.model(states)
            v_targets = self.V_net.model(states)
            v_targets[:, 0] = torch.mul(torch.max(all_q_values_next, axis=1).values, gamma) + rewards
            v_targets[is_dones_indices, 0] = rewards[is_dones_indices]
            v_targets[:, 0] = v_targets[:, 0] - torch.mul(all_q_values[range(len(all_q_values)), actions], self.q_weight)

            self.update_V(states, v_targets)

            v_values_next = self.V_net.model(next_states)
            all_q_values = self.Q_net.model(states) # predicted q_values of all states
            #Update q values
            all_q_values[range(len(all_q_values)), actions] = rewards + torch.mul(v_values_next, gamma)[:, 0]
            all_q_values[is_dones_indices, actions_tensor[is_dones_indices].tolist()] = rewards[is_dones_indices]
                   
            self.update_Q(states, all_q_values)