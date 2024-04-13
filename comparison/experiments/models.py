import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, ReLU, Sequential
from torch.types import _dtype
import torch.nn.functional as F
from typing import List

### BASIC MLP ###
class BasicMLP(nn.Module):
    """
    Basic MLP model with ReLU activation and dropout layers.
    
    Args:
    input_dim (int): Number of input features.
    model_layers (List[int]): List of hidden layer dimensions.
    dropout_rate (float): Dropout rate.
    
    Returns:
    nn.Module: Basic MLP model.
    """
    def __init__(self, input_dim, model_layers, dropout_rate = 0):
        super(BasicMLP, self).__init__()
        layers = []
        prev_layer_dim = input_dim
        for layer_dim in model_layers:
            linear_layer = nn.Linear(prev_layer_dim, layer_dim)

            # He/Kaiming initialization
            nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)

            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_layer_dim = layer_dim

        # Output layer with He/Kaiming initialization
        output_layer = nn.Linear(prev_layer_dim, 1)
        nn.init.kaiming_uniform_(output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

### SAMET'S MODEL ###
class NullAwareFunction(torch.autograd.Function):
    """
    A custom torch function that masks out NaN values during forward propagation 
    and prevents gradient flow through these NaN values during backpropagation.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Masks out NaN values in input.
        """
        ctx.save_for_backward(mask)
        output = input.clone() # Clone the input tensor
        output[mask] = 0.0     # Set NaN values to 0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass. Prevents gradient flow for non-NaN values.
        """
        mask, = ctx.saved_tensors # Retrieve the mask
        grad_output[mask] = 0.0   # Prevent gradient flow for non-NaN values by setting the gradients to 0
        return grad_output, None  # Return the modified gradients and None for the mask
    
class CustomDropoutModel(nn.Module):
    """
    A custom model that first processes NaN values and then processes the rest of the input.

    Args:
    input_layer (int): Number of input features.
    na_layers (List[int]): List of hidden layer dimensions for NaN-aware layers.
    model_layers (List[int]): List of hidden layer dimensions for the main model layers.

    Returns:
    nn.Module: Custom Dropout model.
    """
    
    def __init__(self, input_layer: int, na_layers: List[int], model_layers: List[int]) -> None:
        """
        Initialize the model layers.
        """
        super(CustomDropoutModel, self).__init__()
        
        # Create the NaN-aware layers
        # The last layer is the same size as the input layer
        self.na_layers = self._build_layers(input_layer, na_layers + [input_layer])

        # Create the main model layers
        self.model_layers = self._build_layers(input_layer, model_layers + [1])

    def _build_layers(self, input_layer: int, layers: List[int]) -> nn.ModuleList:
        """
        Helper function to build a series of layers from a given list of layer sizes.
        """
        layers_list = []
        prev_layer = input_layer
        for next_layer in layers:
            linear_layer = nn.Linear(prev_layer, next_layer)
            nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)
            layers_list.append(linear_layer)
            prev_layer = next_layer
        return nn.ModuleList(layers_list)
    
    def forward(self, x: torch.Tensor, return_imputed = False) -> torch.Tensor:
        """
        Forward pass of the model. First processes NaN values, then processes the input.
        """
        # Clone the original input for later use
        original_input = x.clone()
        
        # Mask for NaN values in the input
        null_mask = torch.isnan(x)

        # Process NaN values if any
        if torch.any(null_mask):
            x = NullAwareFunction.apply(x, null_mask)  # Handle NaNs
            for idx, layer in enumerate(self.na_layers):
                x = F.relu(layer(x)) if idx != len(self.na_layers) - 1 else layer(x)
            
            # Update the originally missing input variables with the processed NaN values.
            # Updating the original input in this manner ensures that during backpropagation, only gradients for the 
            # processed NaN values flow backward. The rest of the input doesn't get any gradient update from this operation.
            original_input[null_mask] = x[null_mask]

        # Process the rest of the input
        # Using the original input with update NaN values for the first layer
        for idx, layer in enumerate(self.model_layers):
            x = F.relu(layer(x if idx != 0 else original_input)) if idx != len(self.model_layers) - 1 else layer(x)

        # Return the imputed values alongside the output if required
        if return_imputed:
            return x.squeeze(), original_input
        else:
            return x.squeeze()

### TOMAS' MODEL ###
class NanMask(nn.Module):
    """Create custom learnable mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(NanMask, self).__init__()
        self.mask = torch.isnan(input)
        self.dim = input.shape[1]
        self.W = Linear(self.dim, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.W.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input: Tensor) -> Tensor:
        input[self.mask] = 0
        return self.W(self.mask.float()) + input
    
class MaskedMLP(nn.Module):
    """
    Masked MLP model with custom learnable mask non-linearity.
    
    Args:
    input_dim (int): Number of input features.
    model_layers (List[int]): List of hidden layer dimensions.
    dropout_rate (float): Dropout rate.
    
    Returns:
    nn.Module: Masked MLP model.
    """
    def __init__(self, input_dim, model_layers, dropout_rate):
        super(MaskedMLP, self).__init__()
        self.mlp = BasicMLP(input_dim, model_layers, dropout_rate)
    
    def forward(self, x):
        mask = NanMask(x)
        x = mask(x)
        x = self.mlp(x)
        return x
    
### NEUMISS ###
class Mask(nn.Module):
    """A mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask * input


class SkipConnection(nn.Module):
    """A skip connection operation."""
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMissBlock(nn.Module):
    """The NeuMiss block from "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux."""

    def __init__(self, n_features: int, depth: int,
                 dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        depth : int
            Number of layers (Neumann iterations) in the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)


class NeuMissMLP(nn.Module):
    """A NeuMiss block followed by a MLP."""

    def __init__(self, n_features: int, neumiss_depth: int, mlp_depth: int,
                 mlp_width: int = None, dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs.
        neumiss_depth : int
            Number of layers in the NeuMiss block.
        mlp_depth : int
            Number of hidden layers in the MLP.
        mlp_width : int
            Width of the MLP. If None take mlp_width=n_features. Default: None.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.n_features = n_features
        self.neumiss_depth = neumiss_depth
        self.mlp_depth = mlp_depth
        self.dtype = dtype
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width

        b = int(mlp_depth >= 1)
        last_layer_width = mlp_width if b else n_features
        self.layers = Sequential(
            NeuMissBlock(n_features, neumiss_depth, dtype),
            *[Linear(n_features, mlp_width, dtype=dtype), ReLU()]*b,
            *[Linear(mlp_width, mlp_width, dtype=dtype), ReLU()]*b*(mlp_depth-1),
            *[Linear(last_layer_width, 1, dtype=dtype)],
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out.squeeze()
    



class CustomNeuMissMLP(nn.Module):
    """A NeuMiss block followed by a MLP."""

    def __init__(self, n_features: int, neumiss_depth: int, 
                mlp_layers: List[int], dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs.
        neumiss_depth : int
            Number of layers in the NeuMiss block.
        mlp_layers : List[int]
            Shape of hidden layers in the MLP.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.n_features = n_features
        self.neumiss_depth = neumiss_depth
        self.mlp_layers = mlp_layers
        self.dtype = dtype

        self.layers = [NeuMissBlock(n_features, neumiss_depth, dtype)]

        prev_layer_dim = n_features
        for layer_dim in self.mlp_layers:
            linear_layer = nn.Linear(prev_layer_dim, layer_dim)

            # He/Kaiming initialization
            nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)

            self.layers.append(linear_layer)
            self.layers.append(nn.ReLU())
            prev_layer_dim = layer_dim

        # Output layer with He/Kaiming initialization
        output_layer = nn.Linear(prev_layer_dim, 1)
        nn.init.kaiming_uniform_(output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(output_layer.bias)
        self.layers.append(output_layer)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return out.squeeze()