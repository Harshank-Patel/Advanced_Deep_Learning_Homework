import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder
    """
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        # Define the layers separately for clarity and robustness
        self.patchify = PatchifyLinear(patch_size, latent_dim)
        
        self.encoder_core = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=3, padding=1)
        )
        
        self.decoder_core = torch.nn.Sequential(
            torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=3, padding=1),
            torch.nn.GELU()
        )
        
        self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms.
        """
        encoded_x = self.encode(x)
        decoded_x = self.decode(encoded_x)
        return decoded_x, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Turn image into patches
        x = self.patchify(x)
        
        # Step 2: HWC -> CHW for convolution
        x_chw = hwc_to_chw(x)
        
        # Step 3: Run through the core encoder (GELU, Conv2d)
        encoded_chw = self.encoder_core(x_chw)
        
        # Step 4: CHW -> HWC to return to standard format
        encoded_hwc = chw_to_hwc(encoded_chw)
        return encoded_hwc

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: HWC -> CHW for convolution (FIXED a typo here)
        x_chw = hwc_to_chw(x)
        
        # Step 2: Run through the core decoder (Conv2d, GELU)
        decoded_chw = self.decoder_core(x_chw)
        
        # Step 3: CHW -> HWC
        decoded_hwc = chw_to_hwc(decoded_chw)
        
        # Step 4: Turn patches back into an image
        reconstructed_image = self.unpatchify(decoded_hwc)
        return reconstructed_image