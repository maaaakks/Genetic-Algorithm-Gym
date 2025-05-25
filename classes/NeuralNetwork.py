import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, network_config: dict):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.Sequential()
        current_size = input_size

        activation_mapping = {
            'ReLU': nn.ReLU,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid,
            'LeakyReLU': nn.LeakyReLU,
            # nn.Identity is used for 'Linear' activation
        }

        # Hidden Layers
        if 'hidden_layers' in network_config and network_config['hidden_layers']:
            for i, layer_conf in enumerate(network_config['hidden_layers']):
                neurons = layer_conf['neurons']
                activation_name = layer_conf['activation']
                
                self.layers.add_module(f"hidden_linear_{i}", nn.Linear(current_size, neurons))
                
                if activation_name != 'Linear':
                    activation_fn_class = activation_mapping.get(activation_name)
                    if activation_fn_class:
                        self.layers.add_module(f"hidden_activation_{i}", activation_fn_class())
                    else:
                        # Default to ReLU if activation is not found and not Linear
                        self.layers.add_module(f"hidden_activation_{i}", nn.ReLU())
                current_size = neurons
        
        # Output Layer
        self.layers.add_module("output_linear", nn.Linear(current_size, output_size))
        output_activation_name = network_config.get('output_layer', {}).get('activation', 'Linear')
        
        if output_activation_name != 'Linear':
            output_activation_fn_class = activation_mapping.get(output_activation_name)
            if output_activation_fn_class:
                self.layers.add_module("output_activation", output_activation_fn_class())
            # If output activation is not 'Linear' and not in mapping, no activation is added (effectively linear)


    def forward(self, x):
        return self.layers(x)
