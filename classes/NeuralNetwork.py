#
# You can customize the NN for each environment
#
#

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, env):
        self.actions = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.env_name = str(env.unwrapped.spec.id)
        self.complex_nn_environments = [
            "LunarLander-v2",
            ]
        
        def complex_nn():
            self.layer1 = nn.Linear(self.state_size, 64)
            self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
            self.output_layer = nn.Linear(64, self.actions)

        def simple_nn():
            super(NeuralNetwork, self).__init__()
            self.layer1 = nn.Linear(self.state_size, 16)
            self.layer2 = nn.Linear(16, 16)
            self.output_layer = nn.Linear(16, self.actions)
        
        super(NeuralNetwork, self).__init__()
        if self.env_name in self.complex_nn_environments: complex_nn()
        else:simple_nn()

            
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        if self.env_name in self.complex_nn_environments: x = torch.relu(self.layer3(x))
        x = self.output_layer(x)
        return x