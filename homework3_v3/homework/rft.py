from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_rft_example(question: str, answer: float, reasoning: str) -> dict:
    # Use the full reasoning and answer for training
    return {
        "question": question,
        "answer": reasoning
    }


class TokenizedRFTDataset:
    def __init__(self, tokenizer, data, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted = self.format_fn(*self.data[idx])
        # Use SFT's tokenize function
        from .sft import tokenize
        return tokenize(self.tokenizer, **formatted)


def train_model(
    output_dir: str,
    **kwargs,
):
    # Local imports
    from .base_llm import BaseLLM
    from peft import get_peft_model, LoraConfig
    from transformers import TrainingArguments, Trainer
    import json
    from pathlib import Path

    # Load RFT dataset
    with open(Path(__file__).parent.parent / "data/rft.json", "r") as f:
        rft_data = json.load(f)

    # Initialize base model
    llm = BaseLLM()

    # Allow overriding some training / LoRA params via kwargs for easy tuning
    lora_r = int(kwargs.get("lora_r", 16))
    lora_alpha = int(kwargs.get("lora_alpha", 4 * lora_r))
    use_fp16 = bool(kwargs.get("fp16", False))
    per_device_batch = int(kwargs.get("per_device_train_batch_size", 4))

    # LoRA config (can use higher rank if needed, but keep <50MB)
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=lora_r,
        lora_alpha=lora_alpha,
        bias="none",
        task_type="CAUSAL_LM"
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    if hasattr(llm.model, "config"):
        try:
            llm.model.config.use_cache = False
        except Exception:
            pass

    # Tokenized dataset
    train_dataset = TokenizedRFTDataset(llm.tokenizer, rft_data, format_rft_example)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        # Disable fp16 by default for MPS / macOS. Enable manually on CUDA GPUs.
        fp16=use_fp16,
        # Use conservative dataloader workers on macOS to avoid fork issues.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        learning_rate=2e-4,
        num_train_epochs=int(kwargs.get("num_train_epochs", 8)),
        weight_decay=float(kwargs.get("weight_decay", 0.01)),
        # Smaller default batch size to avoid OOM on memory-constrained devices
        per_device_train_batch_size=per_device_batch,
        save_strategy="epoch",
        save_total_limit=3,
    )

    print("Training with config:")
    print(f"  lora_r={lora_r} lora_alpha={lora_alpha} fp16={use_fp16} per_device_batch={per_device_batch}")
    print(f"  num_train_epochs={training_args.num_train_epochs} weight_decay={training_args.weight_decay}")

    # Trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save adapter to homework/rft_model
    out_path = Path(__file__).parent / "rft_model"
    out_path.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(str(out_path))

    # Test the trained model
    from .sft import test_model
    test_model(str(out_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
