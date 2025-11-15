from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Construct a short chat-style prompt with 1-2 high-quality chain-of-thought
        examples. The examples are concise, show steps, and finish with the
        final answer wrapped in <answer>..</answer>.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that performs unit conversions.\n"
                    "Be concise and show your work step by step (brief arithmetic steps).\n"
                    "Always place the final numeric answer inside <answer>...</answer> with no extra text.\n"
                    "When possible, round the numeric final answer to at most 4 decimal places."
                )
            },
            # One clear example using straightforward steps
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g. 6 * 1000 = <answer>6000</answer>"
            },
            # Second example with slightly different units / steps
            {
                "role": "user",
                "content": "Convert 2 hours into seconds."
            },
            {
                "role": "assistant",
                "content": "1 hour = 3600 seconds. 2 * 3600 = <answer>7200</answer>"
            },
            # The actual question
            {
                "role": "user",
                "content": question
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
