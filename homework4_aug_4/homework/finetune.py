from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import VQADataset, benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")


def load(model_name: str = "vlm_model") -> BaseVLM:
    """
    Load a finetuned VLM with LoRA adapters if available.
    If the directory does not exist or loading fails, fall back to the base VLM.
    """
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    # Start from the base VLM
    vlm = BaseVLM()

    if model_path.exists():
        try:
            print(f"[INFO] Loading LoRA adapters from: {model_path}")
            vlm.model = PeftModel.from_pretrained(vlm.model, model_path).to(vlm.device)
            vlm.model.eval()
        except Exception as e:
            print(f"[WARNING] Failed to load LoRA from {model_path}: {e}")
            print("[WARNING] Falling back to base VLM weights.")
            vlm.model.eval()
    else:
        print(f"[WARNING] LoRA directory {model_path} not found. Using base VLM only.")
        vlm.model.eval()

    return vlm


def custom_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


class VQADatasetForTraining(Dataset):
    def __init__(self, dataset: VQADataset, processor: AutoProcessor):
        self.dataset = dataset
        self.processor = processor
        self.features = ["image", "question", "answer"]
        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        # Prepare input text in chat format
        input_message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        full_text = prompt + item["answer"]  # append the answer to the prompt

        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Get answer length
        answer_ids = self.processor(
            images=None, text=item["answer"], return_tensors="pt", truncation=True
        ).input_ids.squeeze(0)
        answer_len = len(answer_ids)

        # Prepare labels: mask everything except the answer tokens
        labels = input_ids.clone()
        labels[:-answer_len] = -100  # only keep loss on answer

        # Ensure EOS token is at the end of the sequence
        if input_ids[-1] != self.processor.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([self.processor.tokenizer.eos_token_id])])
            attention_mask = torch.cat([attention_mask, torch.tensor([1])])
            labels = torch.cat([labels, torch.tensor([self.processor.tokenizer.eos_token_id])])

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.long(),
        }


def train(
    data_dir: Path | None = None,
    train_dataset_name: str = "train",
    output_dir: str = "vlm_model",
    num_train_epochs: float = 0.1,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-4,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    num_workers: int = 0,
):
    """
    Fine-tune the VLM using LoRA on the generated QA pairs.

    Args:
        data_dir: Root data directory (containing train/valid/train_demo folders)
        train_dataset_name: Split name, e.g. "train" or "train_demo"
        output_dir: Directory (relative to homework/) to save the LoRA adapters
        num_train_epochs: Number of epochs to train
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Grad accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_workers: num_workers for DataLoader (0 is safest on Mac)
    """
    # --- 1. Normalize paths ---
    if data_dir is None:
        # default to homework_root/data
        data_dir = Path(__file__).parent.parent / "data"
    elif isinstance(data_dir, str):
        data_dir = Path(data_dir)

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # --- 2. Base VLM + processor ---
    vlm = BaseVLM()
    model = vlm.model
    proc = vlm.processor

    # --- 3. LoRA config & wrapping ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Some HF models need this for training stability with LoRA
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    model.to(DEVICE)

    # --- 4. Dataset setup ---
    # Try the requested split first
    base_train_dataset = VQADataset(train_dataset_name, data_dir)

    # If no QA pairs for that split, fall back to train_demo
    if len(base_train_dataset) == 0 and train_dataset_name != "train_demo":
        print(
            f"[WARNING] Loaded 0 QA pairs for split='{train_dataset_name}'. "
            "Falling back to 'train_demo' split."
        )
        base_train_dataset = VQADataset("train_demo", data_dir)

    if len(base_train_dataset) == 0:
        raise RuntimeError(
            "No QA pairs found for training. "
            "Ensure you have generated '*_qa_pairs.json' in data/train or data/train_demo."
        )

    train_dataset = VQADatasetForTraining(base_train_dataset, proc)

    # Ensure pad token is set
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token

    # --- 5. TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir),
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True if DEVICE == "cuda" else False,  # bf16 only on CUDA, not MPS
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    # --- 6. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )

    # --- 7. Train & save ---
    trainer.train()
    trainer.save_model(str(output_dir))

    writer.close()

    return model, proc


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    model.train()
    return val_loss / len(val_loader)


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test_model(ckpt_path: str, val_dataset: str = "valid_grader"):
    testset = VQADataset(val_dataset)

    llm = load(ckpt_path)

    benchmark_result = benchmark(llm, testset, 128)
    print(benchmark_result.accuracy)


if __name__ == "__main__":
    from fire import Fire

    Fire({"demo_train": demo_train, "train": train, "test": test_model})
