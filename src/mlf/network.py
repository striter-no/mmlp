import torch
import os
from accelerate import Accelerator
from mlc.network import TransformerLM
from mlf.tokenizer import Tokenizer
from mlf.settings import ModelSettings
from mlc.datasets import Dataset
from mlf.datastorage import DataStorage

import logging
logger = logging.getLogger(__name__)

class Network:
    def __init__(self, tokenizer: Tokenizer, settings: ModelSettings, acc: Accelerator | None = None) -> None:
        mixed_precision = "no"
        if torch.cuda.is_available():
            mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

        if acc is None:
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
        else:
            self.accelerator = acc
        self.device = self.accelerator.device

        self.tokenizer = tokenizer
        self.settings = settings

        self._model = TransformerLM(
            vocab_size=tokenizer.engine.base,
            context_len=settings.context_len,
            embed_dim=settings.embedding_dims,
            hidden=settings.hidden_neurons,
            num_layers=settings.encoder_layers,
            dropout=settings.dropout,
            pad_id=tokenizer.engine.pad_id
        ).to(self.device)

        logger.info(f"[network] using {self.device}")

    @property
    def raw_model(self):
        return self.accelerator.unwrap_model(self._model)

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.raw_model.state_dict(), path)

    @staticmethod
    def load_from_file(model_path: str, settings_path: str, acc: None | Accelerator = None):
        settings = ModelSettings.load_from_file(settings_path)
        dial_ds = Dataset(column_text=settings.column_text, column_title=settings.column_title)
        dial_ds.load(settings.dataset_path, settings.dataset_name)

        storage = DataStorage(cache_path=".cache", dataset=dial_ds)
        storage.load_dialogues(settings.pairs_num)

        tokenizer = Tokenizer(cache_path=".cache", storage=storage)
        tokenizer.load_tokens(settings.tokens_num)

        network = Network(tokenizer=tokenizer, settings=settings, acc=acc)
        network.raw_model.load_state_dict(
            torch.load(model_path, map_location=network.device, weights_only=False)
        )
        network.raw_model.to(network.device)
        network.raw_model.eval()
        return network, settings

    def predict(self, input_text: str, history: list[str] | None = None, temperature=None, top_k=None) -> tuple[str, int, int]:
        temp = self.settings.default_temp if temperature is None else temperature
        topk = self.settings.default_top_k if top_k is None else top_k

        self.raw_model.eval()
        eng = self.tokenizer.engine

        history = history or []

        all_turns = history + [input_text]

        turns_str = []
        for i, text in enumerate(all_turns):
            role = "<user>" if i % 2 == 0 else "<ai>"
            turns_str.append(f"{role} {text}")

        prompt_str = "\n".join(turns_str)

        ids = [eng.bos_id] + self.tokenizer.tokenize(prompt_str) + [eng.ai_id]

        generated = ids[-self.raw_model.context_len:]
        prompt_len = len(generated)
        stop_ids = {eng.eos_id, eng.user_id}

        with torch.no_grad():
            for _ in range(self.raw_model.context_len - prompt_len):
                x = torch.tensor(
                    [generated[-self.raw_model.context_len:]],
                    dtype=torch.long
                ).to(self.device)

                logits = self.raw_model(x)
                next_logits = logits[0, -1, :] / temp

                if topk > 0:
                    top_logits, top_indices = torch.topk(next_logits, topk)
                    probs = torch.softmax(top_logits, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    next_id = top_indices[idx].item()
                else:
                    next_id = next_logits.argmax(-1).item()

                if next_id in stop_ids:
                    break

                generated.append(next_id)

        response_ids = generated[prompt_len:]

        used_context = len(generated)
        max_context = self.raw_model.context_len

        return eng.detokenize(response_ids).strip(), used_context, max_context
