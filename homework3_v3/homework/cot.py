from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that performs unit conversions. Be concise and show your work step by step. Always use the format <answer>value</answer> for your final answer."
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
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
