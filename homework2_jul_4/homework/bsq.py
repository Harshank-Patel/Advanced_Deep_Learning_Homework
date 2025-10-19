import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.projection_down = torch.nn.Linear(embedding_dim, codebook_bits)
        self.projection_up = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Project down to the codebook dimension
        x = self.projection_down(x)
        # L2 normalize along the last (feature) dimension
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # Binarize using the differentiable sign function
        return diff_sign(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.projection_up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        # Use exponentiation instead of bit-shifting for hardware compatibility (Apple Silicon)
        #
        powers_of_2 = 2 ** torch.arange(x.size(-1), device=x.device)
        return (x * powers_of_2).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        # Use exponentiation instead of bit-shifting for hardware compatibility
        powers_of_2 = 2 ** torch.arange(self._codebook_bits, device=x.device)
        binary = ((x.unsqueeze(-1) & powers_of_2) > 0).float()
        return 2 * binary - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
         
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        # Initialize the PatchAutoEncoder part of the model
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        # The embedding dimension for BSQ is the latent_dim of our autoencoder
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First, encode the image into continuous patch embeddings
        continuous_embeddings = super().encode(x)
        # Then, use BSQ to quantize them into integer tokens
        return self.bsq.encode_index(continuous_embeddings)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # First, use BSQ to decode integer tokens into quantized embeddings
        quantized_embeddings = self.bsq.decode_index(x)
        # Then, use the autoencoder's decoder to reconstruct the image from these embeddings
        return super().decode(quantized_embeddings)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # This method uses the original autoencoder's encoder
        return super().encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # This method uses the original autoencoder's decoder
        return super().decode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms.
        """
        # 1. Encode the image into a continuous latent space
        z = self.encode(x)
        # 2. Pass the latent representation through the BSQ module to get a quantized version
        z_quantized = self.bsq(z)
        # 3. Decode the quantized representation back into an image
        x_reconstructed = self.decode(z_quantized)

        # Return the reconstructed image and an empty dictionary for additional losses
        return x_reconstructed, {}