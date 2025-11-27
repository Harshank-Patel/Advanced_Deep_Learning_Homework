from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

# Load the processor used by the base VLM
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    """
    Load a trained CLIP+LoRA model for evaluation.
    This function rebuilds the CLIP architecture and loads the saved LoRA
    adapters and custom projection heads/logit scale.
    """
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    # 1. Rebuild base model components
    vlm = BaseVLM()
    base = CLIP(
        vlm.model.model.vision_model,
        vlm.model.model.text_model
    )
    # Load custom weights (projection heads, logit_scale)
    base.load_pretrained(model_path)

    # 2. Load LoRA adapters onto the base model
    peft_model = PeftModel.from_pretrained(base, model_path)

    # 3. Move to device and dtype
    peft_model = peft_model.to(device)
    if device in ("cuda", "mps"):
        peft_model = peft_model.to(dtype=torch.bfloat16)
    peft_model.eval()

    # 4. Wrapper so grader sees `.model` attribute
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model  # Grader expects the main module here

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    return Wrapper(peft_model)


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training. Pads sequences to the max length
    in the batch. Labels are a placeholder for the Trainer API.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(), # Vision inputs typically use float32/16
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    """Dataset wrapper for preparing image-caption pairs for CLIP training."""
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        # Image augmentation/preprocessing for training stability
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        
        # Tokenize text and append EOS token
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # Placeholder for Trainer API
        }


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) Model.
    Uses the VLM's vision and text backbones with custom projection heads.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Helper to dynamically find the hidden size of the backbones
        def get_hidden_size(module: nn.Module) -> int:
            cfg = getattr(module, "config", None)
            for attr in ["hidden_size", "d_model", "embed_dim"]:
                if cfg is not None and hasattr(cfg, attr):
                    return getattr(cfg, attr)
                if hasattr(module, attr):
                    return getattr(module, attr)
            raise ValueError(f"Cannot infer hidden size for module {module.__class__.__name__}")

        vision_hidden = get_hidden_size(self.vision_encoder)
        text_hidden = get_hidden_size(self.text_encoder)

        # Projection heads: these names MUST NOT contain "lora"
        self.vision_projection = nn.Linear(vision_hidden, proj_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden, proj_dim, bias=False)

        # Learnable logit scale (CLIP-style temperature parameter)
        init_logit_scale = torch.log(torch.tensor(1.0 / temperature))
        self.logit_scale = nn.Parameter(init_logit_scale)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes the image batch into normalized feature vectors.
        """
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        image = image.to(vision_dtype)

        # Forward through vision encoder (extract last hidden state)
        outputs = self.vision_encoder(pixel_values=image)

        # Get the hidden state output (works for most HF models)
        hidden = getattr(outputs, "last_hidden_state", outputs[0])

        # Global average pooling over spatial dimensions/patches (dim=1)
        pooled = hidden.mean(dim=1)  # (B, H_vision)

        # Projection + L2 norm
        feats = self.vision_projection(pooled)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the text batch into normalized feature vectors using masked mean pooling.
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", outputs[0]) # (B, L, H_text)

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom # (B, H_text)

        # Projection + L2 norm
        feats = self.text_projection(pooled)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats

    def save_pretrained(self, save_directory: str, **kwargs):
        """Saves only the custom parameters (projections and logit_scale)."""
        additional_state_dict = {}
        for name, param in self.named_parameters():
            # Exclude backbone weights (which are handled by LoRA)
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Loads only the custom parameters (projections and logit_scale)."""
        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                if name in additional_state_dict:
                    param.data.copy_(additional_state_dict[name])

    def set_trainable_parameters(self):
        """Sets the custom projection heads and logit scale to be trainable."""
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """Enables gradient checkpointing on backbones to save memory."""
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """A necessary step for proper gradient flow when using LoRA/PEFT."""
        def make_inputs_require_grads(module, input, output): # noqa: A002
            output.requires_grad_(True)
        # Apply the hook to the input embeddings of both backbones
        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)


    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None, # ignored, for Trainer API compatibility
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model, computing image and text features,
        and the similarity matrix (logits).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)

        # Compute the similarity matrix (logits)
        logit_scale = self.logit_scale.exp()
        # image_features (B_img, D) @ text_features.t() (D, B_txt) = (B_img, B_txt)
        logits_per_image = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits_per_image


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor, # ignored, for Trainer API compatibility
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute CLIP-style contrastive loss, which is the average of
    Image-to-Text and Text-to-Image cross-entropy losses.
    """
    _, _, logits_per_image = outputs

    batch_size = logits_per_image.size(0)
    device = logits_per_image.device
    
    # Target is the identity matrix, since we expect image[i] to match text[i]
    target = torch.arange(batch_size, dtype=torch.long, device=device)

    # 1. Image-to-Text loss
    loss_i2t = nn.functional.cross_entropy(logits_per_image, target)

    # 2. Text-to-Image loss (use transposed logits)
    loss_t2i = nn.functional.cross_entropy(logits_per_image.t(), target)

    # Average the two losses
    loss = (loss_i2t + loss_t2i) / 2.0
    return loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    """
    Collects the names of all nn.Linear layers within the vision and text
    encoders to be targeted by LoRA, while excluding the projection heads.
    """
    target_modules = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name # Exclude the custom projection heads
        ):
            target_modules.append(name)

    # Use a set to ensure unique module names
    return sorted(list(set(target_modules)))


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip_model",
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 3e-4,
    num_workers: int = 0,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    # ===== 1. Model Setup =====
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    base_clip = CLIP(vision_encoder, text_encoder).to(device)
    # Cast to bfloat16 if using CUDA/MPS for memory efficiency
    if device in ("cuda", "mps"):
        base_clip = base_clip.to(dtype=torch.bfloat16)

    # LoRA setup
    target_modules = get_target_modules_for_lora(base_clip)
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, # Appropriate task type for contrastive learning
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=target_modules, 
        bias="none",
    )
    
    model = get_peft_model(base_clip, peft_config)

    # Make CLIP heads trainable (they are not part of LoRA)
    model.model.set_trainable_parameters()

    # Ensure all LoRA weights are trainable
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = model.to(device)
    
    model.print_trainable_parameters() # Show how many parameters are trainable

    # ===== 2. Dataset Setup =====
    train_caption_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_caption_dataset, processor)

    # ===== 3. Trainer Setup =====
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=num_workers,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
    )

    # Custom compute_loss to use our CLIP contrastive loss, compatible with HF Trainer
    def custom_compute_loss(model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None):
        """
        HuggingFace Trainer will call this with:
            compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)
        We ignore `num_items_in_batch` in the loss itself, but accept it to avoid errors.
        """
        outputs = model(**inputs)
        loss = compute_clip_loss(outputs, inputs.get("labels"), num_items_in_batch=num_items_in_batch)
        if return_outputs:
            return loss, outputs
        return loss

    trainer.compute_loss = custom_compute_loss


    # ===== 4. Train and Save =====
    trainer.train()

    # Save the LoRA adapters (handled by trainer.save_model)
    trainer.save_model(output_dir)
    # Save the custom projection heads and logit scale separately
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    """A small debug run to verify setup."""
    train(
        data_dir=Path("data"),
        output_dir="demo_clip",
        num_train_epochs=0.02,
        per_device_train_batch_size=4,
        num_workers=0,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    """
    Evaluates the trained CLIP model on the multi-choice QA task.
    Uses Image-to-Text retrieval to select the best candidate answer.
    """
    import tqdm

    # Load multi-choice QA dataset
    testset = MultiChoiceQADataset(val_dataset)

    # Load trained CLIP + LoRA
    clip = load(ckpt_path) # Returns the wrapped model for evaluation
    clip.eval()

    # Image processing for evaluation (CenterCrop)
    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        # 1. Image -> tensor
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device)
        if device in ("cuda", "mps"):
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        # 2. Text candidates -> tensors
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # 3. Forward pass
        with torch.no_grad():
            # Use the underlying model directly
            vision_feature, text_feature, _ = clip.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
        # 4. Image-to-Text Retrieval: Choose the text feature most similar to the image feature
        # vision_feature is (1, D), text_feature is (N_candidates, D)
        similarity = torch.matmul(vision_feature, text_feature.T) # (1, N_candidates)
        prediction = similarity.argmax(dim=-1).item()
        
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count:.4f}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test, "demo_train": demo_train})


if __name__ == "__main__":
    main()