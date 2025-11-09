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

    # LoRA config (can use higher rank if needed, but keep <50MB)
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=32,  # Higher rank for RFT, adjust if needed
        lora_alpha=128,  # alpha = 4*r
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
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_strategy="epoch",
        save_total_limit=3,
    )

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
