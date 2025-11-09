import json
import re
from pathlib import Path
from .cot import CoTModel
from .data import Dataset


def extract_answer(text):
    # Extract float from <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.7):
    # Load train data
    train_data = Dataset("train")
    cot_model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    results = []
    for question, gt_answer in train_data:
        prompt = cot_model.format_prompt(question)
        # Generate multiple completions
        generations = cot_model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature
        )
        # Flatten if needed
        if isinstance(generations[0], list):
            generations = generations[0]
        found = False
        for gen in generations:
            pred = extract_answer(gen)
            # Accept if answer matches ground truth (allow small tolerance)
            if pred is not None and abs(pred - float(gt_answer)) < 1e-2:
                results.append([question, gt_answer, gen])
                found = True
                break  # Only keep one correct reasoning per question
        # Optionally, print if no correct answer found
        # if not found:
        #     print(f"No correct answer for: {question}")
    # Save to rft.json
    out_path = Path(__file__).parent.parent / output_json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Generated {len(results)} RFT examples to {out_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)
