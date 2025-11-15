import json
import re
import time
from pathlib import Path
from typing import List

from .cot_fixed import CoTModel
from .data import Dataset


def extract_answer(text: str):
    # Extract float from <answer>...</answer>
    if not isinstance(text, str):
        return None
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _group_generations(generations: List[str], batch_size: int, num_return: int):
    """Group flat list of generations into per-question lists.

    Some batched_generate implementations return a flat list of length
    batch_size * num_return. Others return a nested list (one list per prompt).
    This helper normalizes to a list-of-lists.
    """
    if len(generations) == 0:
        return [[] for _ in range(batch_size)]

    # If already nested (list of lists), return as-is (ensure length matches)
    if isinstance(generations[0], list):
        return generations

    # Otherwise assume flat list and chunk it
    grouped = []
    for i in range(batch_size):
        start = i * num_return
        end = start + num_return
        grouped.append(generations[start:end])
    return grouped


def generate_dataset(
    output_json: str = "data/rft.json",
    oversample: int = 20,
    temperature: float = 0.8,
    batch_size: int = 4,
    model_checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    max_examples: int = 150,
    target_rate: float = 0.8,
    max_variants_per_question: int = 3,
    rounding: int = 4,
):
    """
    Generate an RFT dataset using rejection sampling from an in-context CoT model.

    - output_json: path (relative to repo root) where rft json will be written
    - oversample: number of completions per question
    - temperature: sampling temperature
    - batch_size: how many questions to send to the model at once
    - model_checkpoint: which model checkpoint to load for rollouts
    - max_examples: maximum number of questions to process (useful for debugging)
    - target_rate: stop early if we've collected >= target_rate * max_examples valid examples
    """
    train_data = Dataset("train")

    # Collect up to max_examples questions
    questions = train_data[:max_examples] if hasattr(train_data, "__getitem__") else list(train_data)[:max_examples]

    cot_model = CoTModel(checkpoint=model_checkpoint)

    results = []
    total = len(questions)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]
        prompts = [cot_model.format_prompt(q) for q, _ in batch]

        # Call batched_generate once per batch to utilize accelerator efficiently.
        # If we run out-of-memory (common on MPS / small GPUs), fall back to
        # smaller micro-batches per prompt to avoid OOM.
        try:
            generations = cot_model.batched_generate(
                prompts,
                num_return_sequences=oversample,
                temperature=temperature,
            )
            # Normalize to list-of-lists: per-question generations
            grouped = _group_generations(generations, len(prompts), oversample)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps" in msg or "cuda out of memory" in msg:
                print("OOM detected during batched generation; falling back to micro-batch generation.")
                # Micro-generate per prompt in this batch to avoid OOM.
                grouped = []
                for p in prompts:
                    # accumulate generations for this prompt in small chunks
                    acc = []
                    remaining = oversample
                    chunk = min(4, oversample)
                    while remaining > 0:
                        try:
                            gen_chunk = cot_model.batched_generate(
                                [p],
                                num_return_sequences=min(chunk, remaining),
                                temperature=temperature,
                            )
                        except RuntimeError as e2:
                            # If still OOM on single prompt, reduce chunk and retry
                            if chunk <= 1:
                                raise
                            chunk = max(1, chunk // 2)
                            continue
                        # batched_generate may return nested list for single prompt
                        if isinstance(gen_chunk, list) and len(gen_chunk) > 0 and isinstance(gen_chunk[0], list):
                            gen_list = gen_chunk[0]
                        else:
                            # flat list
                            gen_list = gen_chunk
                        acc.extend(gen_list)
                        remaining = oversample - len(acc)
                    grouped.append(acc[:oversample])
            else:
                # re-raise unexpected errors
                raise

        def _normalize_reasoning(reasoning: str, answer_value: float) -> str:
            """Replace the numeric inside <answer>...</answer> with a rounded value.

            This avoids tiny floating point mismatches between model output and
            the dataset ground-truth, and standardizes training targets.
            """
            fmt = f"{answer_value:.{rounding}f}"
            if isinstance(reasoning, str):
                if re.search(r"<answer>.*?</answer>", reasoning):
                    return re.sub(r"<answer>.*?</answer>", f"<answer>{fmt}</answer>", reasoning)
                else:
                    return reasoning + f" <answer>{fmt}</answer>"
            return reasoning

        # Iterate through each question's generated outputs and collect up to
        # `max_variants_per_question` distinct correct reasoning variants.
        for (question, gt_answer), gens in zip(batch, grouped):
            collected_variants = 0
            seen_texts = set()
            for gen in gens:
                pred = extract_answer(gen)
                if pred is not None and abs(pred - float(gt_answer)) < 1e-2:
                    # deduplicate identical reasoning strings
                    if gen in seen_texts:
                        continue
                    # normalize the numeric inside <answer> to consistent precision
                    gen_norm = _normalize_reasoning(gen, float(gt_answer))
                    results.append([question, float(round(gt_answer, rounding)), gen_norm])
                    seen_texts.add(gen)
                    collected_variants += 1
                    if collected_variants >= max_variants_per_question:
                        break
            # optional: you could collect near-misses or top candidates here

            # If we didn't find any correct generation, try a deterministic fallback
            # (single greedy generation at temperature 0) and accept it if correct.
            if collected_variants == 0:
                try:
                    fallback = cot_model.batched_generate([prompts[0]], num_return_sequences=1, temperature=0.0)
                    # batched_generate may return nested lists
                    if isinstance(fallback, list) and len(fallback) > 0 and isinstance(fallback[0], list):
                        fallback_text = fallback[0][0]
                    elif isinstance(fallback, list):
                        fallback_text = fallback[0]
                    else:
                        fallback_text = str(fallback)
                    pred_fb = extract_answer(fallback_text)
                    if pred_fb is not None and abs(pred_fb - float(gt_answer)) < 1e-2:
                        fb_norm = _normalize_reasoning(fallback_text, float(gt_answer))
                        results.append([question, float(round(gt_answer, rounding)), fb_norm])
                except RuntimeError:
                    # ignore fallback OOM or other generation errors
                    pass

        # Progress print
        elapsed = time.time() - start_time
        collected = len(results)
        pct = collected / max(1, min(max_examples, total))
        print(f"Processed {min(i+batch_size, total)}/{total} questions, collected={collected}, elapsed={elapsed:.1f}s, rate={pct:.2%}")

        # Stop early if we've reached target coverage
        if collected >= target_rate * min(max_examples, total):
            print(f"Reached target rate {target_rate:.2%} with {collected} examples, stopping early.")
            break

    # Save to output_json (path relative to repo root)
    out_path = Path(__file__).parent.parent / output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} RFT examples to {out_path} in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
