from collections import deque
import random

import torch

class ReplayMemory:
    def __init__(self,max_len,seed =None):
        self.memory = deque([],maxlen=max_len)

        if seed is not None:
            random.seed(seed)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Unzip the batch
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.memory)
