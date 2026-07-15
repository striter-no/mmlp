import torch
import time

from attr import dataclass
from mlc.network import text_to_ids
from mlf.datastorage import DataStorage
from mlf.network import Network
from mlf.tokenizer import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class EpochInfo:
    time_elapsed: float
    total_loss: float
    error: float

class Training:
    def __init__(
        self,
        storage: DataStorage,
        tokenizer: Tokenizer,
        model: Network
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.storage = storage

        self.training_data = []
        self.batch_size = -1
        self.learning_rate = -1

    def show_info(self):
        info = "[training] general info:\n"
        info += " - network info:\n"
        info += f"   - hidden neurons: {self.model.hidden_num}\n"
        info += f"   - embed dimentions: {self.model.embed_dims}\n"
        info += f"   - context length: {self.model.context_len}\n"
        info += f"   - dropout: {self.model.dropout}\n"
        info += f"   - using device: {self.model.device}\n"
        info += " - training at:\n"
        info += f"   - batch size: {self.batch_size}\n"
        info += f"   - learning rate: {self.learning_rate}\n"
        info += f" - loaded tokens: {len(self.tokenizer.tokens)}\n"
        info += f" - loaded Q/A pairs: {len(self.storage.pairs)}\n"

        print(info)

    def prepare_data(
        self,
        batch_size: int = 64,
        learning_rate = 0.001
    ):
        for q_text, a_text in self.storage.pairs:
            q_ids = torch.tensor(
                text_to_ids(
                    self.tokenizer.engine,
                    self.tokenizer.filter_text(q_text),
                    self.model._model.context_len
                ),
                dtype=torch.long
            )

            a_ids = text_to_ids(
                self.tokenizer.engine,
                a_text,
                self.model._model.context_len
            )
            self.training_data.append((q_ids, torch.tensor(a_ids, dtype=torch.long)))

        X = torch.stack([q for q, _ in self.training_data])
        Y = torch.stack([a for _, a in self.training_data])  # (N, context_max_tokens), long

        dataset = TensorDataset(X, Y)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model._model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.engine.pad_id
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_epoch(self) -> EpochInfo:
        if len(self.training_data) == 0:
            raise RuntimeError("Cannot train without training data")

        self.model._model.train()

        start = time.time()
        total_loss = 0

        for batch_X, batch_Y in self.dataloader:
            batch_X, batch_Y = batch_X.to(self.model.device), batch_Y.to(self.model.device)
            self.optimizer.zero_grad()

            pred_logits = self.model._model(batch_X, batch_Y)
            loss = self.loss_fn(pred_logits.transpose(1, 2), batch_Y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model._model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        got = time.time() - start
        return EpochInfo(
            time_elapsed=got,
            total_loss=total_loss,
            error=total_loss / len(self.dataloader)
        )
