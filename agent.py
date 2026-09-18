# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.optim as optim
import random
from dqn import DQN 
from replay_memory import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent:
    def __init__(self,state_dim,action_dim,lr=1e-3,gamma=0.99):
        self.device = device
        self.q_network = DQN(state_dim,
            action_dim
        ).to(self.device)

        self.target_network = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=lr
        )

        self.gamma = gamma
        self.epsilon = 1.0
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random()<self.epsilon:
            return random.randint(0,self.action_dim-1)
        with torch.no_grad():
            state_tensor = torch.tensor(state,
            dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            q_value = self.q_network(state_tensor)
            return q_value.argmax(dim=1).item()

    def train_step(self,batch):
        states, actions, rewards, next_states, terminated = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminated = terminated.to(self.device)

        q_values = self.q_network(states)

        q_selected = q_values.gather(1,actions.unsqueeze(1))
        with torch.no_grad():
            next_q_value = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards+self.gamma*next_q_value*(~terminated).float()
        
        loss = F.mse_loss(
            q_selected.squeeze(),
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        ) 
    def epsilon_decay(self,decay_rate=0.9995,epsilon_min=0.05):
        self.epsilon = max(epsilon_min,self.epsilon*decay_rate)
    
        