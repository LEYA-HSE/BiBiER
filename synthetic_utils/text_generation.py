import logging
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

class TextGenerator:
    def __init__(
        self,
        model_name="gpt2",
        device="cuda",
        max_new_tokens=50,
        temperature=1.0,
        top_p=0.95,
        seed=None
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        logging.info(f"[TextGenerator] Загрузка модели {model_name} на {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if seed is not None:
            set_seed(seed)
            logging.info(f"[TextGenerator] Сид генерации установлен через transformers.set_seed({seed})")
        else:
            logging.info("[TextGenerator] Сид генерации не установлен (seed=None)")

        # --- Примеры для few-shot обучения ---
        self.fewshot_examples = [
            ("happy", "We finally made it!", "We finally made it! I’ve never felt so alive and proud of what we accomplished."),
            ("sad", "He didn't come back.", "He didn't come back. I waited all night, hoping to see him again."),
            ("anger", "Why would you do that?", "Why would you do that? You had no right to interfere!"),
            ("fear", "Did you hear that?", "Did you hear that? Something’s moving outside the window..."),
            ("surprise", "Oh wow, really?", "Oh wow, really? I didn’t see that coming at all!"),
            ("disgust", "That smell is awful.", "That smell is awful. I feel like I’m going to be sick."),
            ("neutral", "Let's meet at noon.", "Let's meet at noon. We’ll have plenty of time to talk then.")
        ]

    def build_prompt(self, emotion: str, partial_text: str) -> str:
        few_shot = random.sample(self.fewshot_examples, 2)
        examples_str = ""
        for emo, text, cont in few_shot:
            examples_str += (
                f"Example:\n"
                f"Emotion: {emo}\n"
                f"Text: {text}\n"
                f"Continuation: {cont}\n\n"
            )

        prompt = (
            "You are a helpful assistant that generates emotionally-aligned sentence continuations.\n"
            "You must include the original sentence in the output, and then continue it in a fluent and emotionally appropriate way.\n\n"
            f"{examples_str}"
            f"Now try:\n"
            f"Emotion: {emotion}\n"
            f"Text: {partial_text}\n"
            f"Continuation:"
        )
        return prompt

    def generate_text(self, emotion: str, partial_text: str = "") -> str:
        prompt = self.build_prompt(emotion, partial_text)
        logging.debug(f"[TextGenerator] prompt:\n{prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.debug(f"[TextGenerator] decoded:\n{full_text}")

        # Вытаскиваем то, что идёт после последнего "Continuation:"
        if "Continuation:" in full_text:
            result = full_text.split("Continuation:")[-1].strip()
        else:
            result = full_text.strip()

        result = result.split("\n")[0].strip()
        return result
