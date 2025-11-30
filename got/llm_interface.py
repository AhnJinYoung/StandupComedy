from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random


class LLMInterface(ABC):
    """Abstract interface for the LLM."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generates text based on the prompt."""
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Returns a mock response."""
        if "Category:" in prompt and "Reason:" in prompt:
            return "Category: punchline\nReason: The tension is set; time to pay it off."
        if "Next:" in prompt and "Pros:" in prompt:
            return (
                "Score: 7\n"
                "Pros: Relatable, clear visual.\n"
                "Cons: Could use sharper twist.\n"
                "Reasoning: Solid beat with room to heighten.\n"
                "Next: branch_punchline"
            )
        if "Score:" in prompt or "Rate the following" in prompt:
            return f"Score: {random.randint(6, 9)}\nReasoning: This is a funny joke with good structure."
        elif "Setup:" in prompt and "punchline" not in prompt.lower():
            return f"Setup: Why did the programmer quit his job? {random.randint(1, 100)}"
        elif "Punchline:" in prompt:
            return f"Punchline: Because he didn't get arrays! {random.randint(1, 100)}"
        elif "Incongruity:" in prompt:
            return "Incongruity: But the vending machine demanded a cover letter."
        elif "Callback:" in prompt:
            return "Callback: Remember that vending machine? It just asked for health insurance."
        else:
            return "Mock response"


class LlamaLLM(LLMInterface):
    """LLM implementation using transformers for Llama 3.1 and the comedy adapter."""

    def __init__(
        self,
        model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_path: Optional[str] = "napalna/Llama-3.1-Comedy-Adapter-Lables",
        quantization: str = "4bit",
        device: str = "auto",
        system_prompt: str = "You are a professional stand-up comedian. Keep outputs in the exact format requested. Be concise and witty.",
    ):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("Please install transformers, torch, and bitsandbytes to use LlamaLLM.")

        self.system_prompt = system_prompt

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"Loading base model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
        )

        if adapter_path:
            try:
                from peft import PeftModel
                print(f"Loading comedy adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                raise ImportError("Please install peft to use LoRA adapters.")

        print("Model ready for inference.")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        from torch import ones_like

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        attention_mask = ones_like(input_ids)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
