from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # LoRA adapter layers
        # The 'A' matrix is a down-projection from in_features to lora_dim.
        # It is initialized with a standard normal distribution.
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)

        # The 'B' matrix is an up-projection from lora_dim to out_features.
        # It is initialized to all zeros. This ensures that the adapter
        # adds zero to the output initially, and training starts from the
        # pretrained state.
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)
        self.lora_b.weight.data.zero_()

        # The parent HalfLinear layer is already not trainable due to the call
        # to self.requires_grad_(False) in its __init__.
        # The newly created LoRA layers are trainable by default.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for the original half-precision linear layer
        linear_output = super().forward(x)
        
        # Forward pass for the LoRA adapter.
        # The inputs to the LoRA layers should be float32, which x is.
        lora_output = self.lora_b(self.lora_a(x))

        # Add the outputs of the linear layer and the LoRA adapter
        return linear_output + lora_output


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim=lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim=lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim=lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net