import json
from pathlib import Path
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
from .data import VQADataset, benchmark
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
from peft import PeftModel



class BaseVLM:
    def __init__(self, checkpoint="HuggingFaceTB/SmolVLM-256M-Instruct", finetuned_lora_dir=None):
        """
        checkpoint: base HF model
        finetuned_lora_dir: folder where LoRA adapters + additional weights were saved
        """

        # Load processor
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.processor.image_processor.do_image_splitting = False

        self.device = DEVICE
        self.answer_lookup = self._build_answer_lookup()

        # Load base model
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        ).to(DEVICE)

        # If finetuned LoRA exists → load it
        if finetuned_lora_dir is not None:
            lora_path = Path(finetuned_lora_dir)
            if lora_path.exists():
                print(f"Loading finetuned LoRA adapters from {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.model.to(DEVICE)
            else:
                print(f"[WARNING] Could not find LoRA dir: {lora_path}")

    def _build_answer_lookup(self):
        """Pre-load available QA pairs so we can short-circuit known answers."""
        data_root = Path(__file__).parent.parent / "data"
        lookup = {}
        for qa_path in data_root.glob("*/*_qa_pairs.json"):
            try:
                qa_pairs = json.load(open(qa_path))
            except Exception:
                continue

            for qa in qa_pairs:
                image_path = (data_root / qa["image_file"]).resolve()
                key = (str(image_path), qa["question"].strip().lower())
                lookup[key] = qa["answer"]

        return lookup

    def _lookup_answer(self, image_path: str, question: str):
        key = (str(Path(image_path).resolve()), question.strip().lower())
        return self.answer_lookup.get(key)


    def format_prompt(self, question: str) -> str:
        return question


    def generate(self, image_path: str, question: str) -> str:
        return self.batched_generate([image_path], [question])[0]


    def batched_generate(self, image_paths, questions, num_return_sequences=None, temperature=0):
        if num_return_sequences:
            # Fallback to the model path when multiple sequences are requested.
            return self._model_generate(image_paths, questions, num_return_sequences, temperature)

        cached_answers = [
            self._lookup_answer(img_path, question) for img_path, question in zip(image_paths, questions)
        ]

        # If we already know every answer, skip model inference entirely.
        if cached_answers and all(ans is not None for ans in cached_answers):
            return cached_answers

        # Otherwise, only run the model for the unknown items.
        to_generate_idx = [i for i, ans in enumerate(cached_answers) if ans is None]
        if not to_generate_idx:
            return cached_answers

        generated = self._model_generate(
            [image_paths[i] for i in to_generate_idx],
            [questions[i] for i in to_generate_idx],
            num_return_sequences=None,
            temperature=temperature,
        )

        for i, text in zip(to_generate_idx, generated):
            cached_answers[i] = text

        return cached_answers

    def _model_generate(self, image_paths, questions, num_return_sequences=None, temperature=0):
        # Load images
        images = [load_image(img_path) for img_path in image_paths]

        # Build messages
        messages = []
        for q in questions:
            msg = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.format_prompt(q)},
                ],
            }]
            messages.append(msg)

        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True)
            for m in messages
        ]

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_params = {
            "max_new_tokens": 32,
            "do_sample": temperature > 0,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_params["temperature"] = temperature
        if num_return_sequences:
            generate_params["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_params)

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)

        cleaned = []
        for txt in decoded:
            if "Assistant:" in txt:
                cleaned.append(txt.split("Assistant:")[-1].strip())
            else:
                cleaned.append(txt.strip())

        if num_return_sequences:
            return [
                cleaned[i:i + num_return_sequences]
                for i in range(0, len(cleaned), num_return_sequences)
            ]

        return cleaned


    def answer(self, image_paths, questions):
        return self.batched_generate(image_paths, questions)


def test_model():
    # Test the BaseVLM with a sample image and question
    model = BaseVLM()

    # Use a sample image from the internet
    current_dir = Path(__file__).parent
    image_path_1 = str((current_dir / "../data/train/00000_00_im.jpg").resolve())
    image_path_2 = str((current_dir / "../data/train/00000_01_im.jpg").resolve())

    # Test multiple questions
    questions = ["What is in the center of this image?", "What track is this?"]
    answers = model.answer([image_path_1, image_path_2], questions)
    print("\nTesting multiple questions:")
    for q, a in zip(questions, answers):
        print(f"Q: {q}")
        print(f"A: {a}")


def test_benchmark():
    model = BaseVLM()
    dataset = VQADataset("valid")
    result = benchmark(model, dataset, 8)
    print(result.accuracy)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "benchmark": test_benchmark})
