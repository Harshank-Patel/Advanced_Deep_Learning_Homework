import os

# Avoid tokenizer parallelism warnings when the process gets forked by the
# DataLoader. Set this early before any tokenizers are imported/used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round answer to reasonable precision to make it easier for the model
    rounded_answer = f"{float(answer):.4f}"
    
    return {
        "question": prompt,
        "answer": f"<answer>{rounded_answer}</answer>"
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    """
    Train the model using LoRA adapter.
    Args:
        output_dir: Directory to save model checkpoints
    """
    # Initialize base model
    llm = BaseLLM()
    
    # Get training dataset
    trainset = Dataset("train")
    
    # Configure LoRA adapter
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=16,  # Rank for adapter, adjust to stay under 20MB
        lora_alpha=64,  # alpha = 4*r
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create LoRA model
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()

    # When using gradient checkpointing, generation cache must be disabled.
    # Explicitly set use_cache=False on the model config to avoid Trainer warnings
    # and ensure correct memory behaviour.
    if hasattr(llm.model, "config"):
        try:
            llm.model.config.use_cache = False
        except Exception:
            # Non-fatal: some wrappers may not expose config in the same way
            pass
    
    # Create tokenized dataset
    train_dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)
    
    # Configure training arguments
    from transformers import TrainingArguments
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
    
    # Create and run trainer
    from transformers import Trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Ensure output directory exists and save final LoRA adapter there.
    from pathlib import Path
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save the PEFT adapter and any related config files to the output dir.
    llm.model.save_pretrained(str(out_path))

    # Test the trained model using the same output directory (local path).
    test_model(str(out_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    # Prefer local checkpoint directories. If the provided path doesn't exist or
    # doesn't contain the adapter config, try to find the latest `checkpoint-*`
    # inside the provided run directory. As a final fallback, try
    # `homework/sft_model`.
    from pathlib import Path
    ckpt = Path(ckpt_path)

    alt = Path(__file__).parent / "sft_model"

    # If ckpt doesn't exist, try alt
    if not ckpt.exists():
        if alt.exists():
            print(f"Warning: checkpoint '{ckpt_path}' not found. Falling back to '{alt}'.")
            ckpt = alt
        else:
            raise ValueError(
                f"Checkpoint path '{ckpt_path}' does not exist and no fallback found at '{alt}'.\n"
                "Make sure you pass the local output_dir used during training or save the adapter with `llm.model.save_pretrained(output_dir)`"
            )

    # If the provided path is a run folder (contains checkpoint-*), try to find the
    # latest checkpoint that contains adapter files.
    if not (ckpt / "adapter_config.json").exists():
        # search for checkpoint-* subdirs
        checkpoints = sorted([d for d in ckpt.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
        if checkpoints:
            latest = checkpoints[-1]
            if (latest / "adapter_config.json").exists():
                print(f"Using latest checkpoint '{latest}' inside '{ckpt}'.")
                ckpt = latest
        # if still not found, fallback to alt
    if not (ckpt / "adapter_config.json").exists():
        if alt.exists() and (alt / "adapter_config.json").exists():
            print(f"Falling back to local adapter at '{alt}'.")
            ckpt = alt
        else:
            raise ValueError(
                f"Adapter config not found in '{ckpt}' or any checkpoints inside it.\n"
                "Please ensure the PEFT adapter was saved locally (e.g. with `llm.model.save_pretrained(output_dir)`)"
            )

    llm.model = PeftModel.from_pretrained(llm.model, str(ckpt)).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
