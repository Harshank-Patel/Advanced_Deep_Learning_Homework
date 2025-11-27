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

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    """
    Load a trained CLIP+LoRA model for evaluation.
    Compatible with both: your test() and the provided grader.
    """
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    # 1. Rebuild base model
    vlm = BaseVLM()
    base = CLIP(
        vlm.model.model.vision_model,
        vlm.model.model.text_model
    )
    base.load_pretrained(model_path)

    # 2. Load LoRA adapters
    peft_model = PeftModel.from_pretrained(base, model_path)

    # 3. Move to device and dtype
    peft_model = peft_model.to(device)
    if device in ("cuda", "mps"):
        peft_model = peft_model.to(dtype=torch.bfloat16)
    peft_model.eval()

    # 4. Wrapper so grader sees `.model`
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model  # grader will use this

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    return Wrapper(peft_model)


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
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
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
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

        # --- helper for hidden sizes (robust to different HF configs) ---
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

        # Projection heads (names include "projection" so LoRA excludes them)
        self.vision_projection = nn.Linear(vision_hidden, proj_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden, proj_dim, bias=False)

        # Learnable logit scale (CLIP-style)
        # temperature ~ 0.07 => logit_scale ~ log(1/0.07)
        init_logit_scale = torch.log(torch.tensor(1.0 / temperature))
        self.logit_scale = nn.Parameter(init_logit_scale)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, H, W)
        returns: (B, D) normalized image features
        """
        # Make sure image dtype matches the vision encoder weights (important on MPS)
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        image = image.to(vision_dtype)

        # Forward through vision encoder
        outputs = self.vision_encoder(pixel_values=image)

        # last_hidden_state: (B, N, H)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]

        # Global average pooling over patches
        pooled = hidden.mean(dim=1)  # (B, H)

        # Projection + L2 norm
        feats = self.vision_projection(pooled)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        returns: (B, D) normalized text features
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # (B, L, H)
        else:
            hidden = outputs[0]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom

        feats = self.text_projection(pooled)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")


    # def encode_image(self, image: torch.Tensor) -> torch.Tensor:
    #     return self.vision_encoder(image)

    # def encode_text(self, text: str) -> torch.Tensor:
    #     return self.text_encoder(text)

    # def save_pretrained(self, save_directory: str, **kwargs):
    #     """Customize save method, save additional parameters"""

    #     additional_state_dict = {}
    #     for name, param in self.named_parameters():
    #         if "vision_encoder." in name or "text_encoder." in name:
    #             continue
    #         additional_state_dict[name] = param.data

    #     torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.

        Returns:
            image_features: (B_img, D)
            text_features: (B_txt, D)
            logits_per_image: (B_img, B_txt) similarity matrix
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)

        # Similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits_per_image


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute CLIP-style contrastive loss.

    outputs: (image_features, text_features, logits_per_image)
    labels: ignored (needed only for Trainer API compatibility)
    """
    image_features, text_features, logits_per_image = outputs

    # images -> texts
    batch_size = image_features.size(0)
    device = image_features.device
    target = torch.arange(batch_size, dtype=torch.long, device=device)

    loss_i2t = nn.functional.cross_entropy(logits_per_image, target)

    # texts -> images
    loss_t2i = nn.functional.cross_entropy(logits_per_image.t(), target)

    loss = (loss_i2t + loss_t2i) / 2.0
    return loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and ("vision_encoder" in name and "projection" not in name):
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


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

    # ===== model =====
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    base_clip = CLIP(vision_encoder, text_encoder).to(device)
    if device in ("cuda", "mps"):
        base_clip = base_clip.to(dtype=torch.bfloat16)

    # LoRA setup
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(base_clip),
        bias="none",
    )
    model = get_peft_model(base_clip, peft_config)

    # Make CLIP heads trainable
    model.model.set_trainable_parameters()

    # Ensure LoRA weights are trainable too
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = model.to(device)

    # ===== dataset =====
    train_caption_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_caption_dataset, processor)

    # ===== trainer args =====
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
        compute_loss_func=compute_clip_loss,
    )

    # ===== train =====
    trainer.train()

    # ===== save final model =====
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    # tiny debug run
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
    import tqdm

    # Load QA dataset
    testset = MultiChoiceQADataset(val_dataset)

    # Load trained CLIP + LoRA
    clip = load(ckpt_path)   # returns a PeftModel
    clip.eval()

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
        # Image -> tensor
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device)
        if device in ("cuda", "mps"):
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        # Text candidates -> tensors
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward pass with **keyword arguments**
        with torch.no_grad():
            vision_feature, text_feature, _ = clip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Choose the most similar text
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction.item() == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count:.4f}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
