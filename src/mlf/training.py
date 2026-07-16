import torch
import time
from attr import dataclass
from mlf.datastorage import DataStorage
from mlf.network import Network
from mlf.tokenizer import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

import logging
logger = logging.getLogger(__name__)

@dataclass
class EpochInfo:
    time_elapsed: float
    total_loss: float
    error: float

class Training:
    def __init__(self, storage: DataStorage, tokenizer: Tokenizer, model: Network):
        self.tokenizer = tokenizer
        self.model = model
        self.storage = storage
        self.dataloader = None
        self.batch_size = -1
        self.learning_rate = -1
        self.accelerator = model.accelerator

    def show_info(self):
        s = self.model.settings
        info = "[training] general info:\n"
        info += f" - network info:\n   - hidden (FFN): {s.hidden_neurons}\n   - embed: {s.embedding_dims}\n   - context: {s.context_len}\n"
        info += f" - training at:\n   - batch: {self.batch_size}\n   - lr: {self.learning_rate}\n"
        info += f" - loaded dialogues: {len(self.storage.dialogues)}\n"
        info += f" - device: {self.accelerator.device}\n"
        info += f" - mixed precision: {self.accelerator.mixed_precision}\n"
        self.accelerator.print(info)

    def prepare_data(self, batch_size=64, learning_rate=0.001, epochs=100, start_epoch=0):
        X_list = []
        for dialog in self.storage.dialogues:
            ids = self.tokenizer.tokenize(dialog)
            if len(ids) > self.model.settings.context_len:
                ids = ids[:self.model.settings.context_len]
            if len(ids) < self.model.settings.context_len:
                ids = ids + [self.tokenizer.engine.pad_id] * (self.model.settings.context_len - len(ids))
            X_list.append(ids)

        X = torch.tensor(X_list, dtype=torch.long)
        Y = torch.cat([X[:, 1:], torch.full((X.size(0), 1), self.tokenizer.engine.pad_id, dtype=torch.long)], dim=1)

        dataset = TensorDataset(X, Y)

        global_batch_size = batch_size * self.accelerator.num_processes
        self.dataloader = DataLoader(dataset, batch_size=global_batch_size, shuffle=True)

        self.optimizer = torch.optim.AdamW(self.model._model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.engine.pad_id)

        steps_per_epoch = len(self.dataloader)
        # total_steps = steps_per_epoch * epochs

        start_step = steps_per_epoch * start_epoch

        if start_step > 0:
            for group in self.optimizer.param_groups:
                group['initial_lr'] = learning_rate

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=1e-5
            )

        self.model._model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model._model, self.optimizer, self.dataloader
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train_epoch(self) -> EpochInfo:
        if self.dataloader is None:
            raise RuntimeError("No data")

        self.model._model.train()
        start = time.time()
        total_loss = 0

        for batch_X, batch_Y in self.dataloader:
            self.optimizer.zero_grad()

            logits = self.model._model(batch_X)
            loss = self.loss_fn(logits.transpose(1, 2), batch_Y)

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model._model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        got = time.time() - start
        self.accelerator.wait_for_everyone()

        avg_loss = total_loss / len(self.dataloader)

        if self.accelerator.is_main_process:
            self.scheduler.step(avg_loss)

        return EpochInfo(time_elapsed=got, total_loss=total_loss, error=avg_loss)
