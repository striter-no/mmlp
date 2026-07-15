from mlc.network import AutoregrssionNetwork, text_to_ids
from mlf.tokenizer import Tokenizer
import torch


class Network:
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_dims = 128,
        hidden_neurons = 256,
        context_len = 50,
        dropout = 0.3,
        device: str = "auto"
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.hidden_num = hidden_neurons
        self.embed_dims = embedding_dims
        self.dropout = dropout
        self.context_len = context_len

        self.tokenizer = tokenizer
        self._model = AutoregrssionNetwork(
            context_len=context_len,
            dropout=dropout,
            embed_dim=embedding_dims,
            hidden=hidden_neurons,
            pad_id=tokenizer.engine.pad_id,
            use_attention=True,
            vocab_size=tokenizer.engine.base
        ).to(self.device)

    def load_from_file(self, path: str):
        self._model.load(path, self.device)

    def save_to_file(self, path: str):
        self._model.save(path)

    def predict(self, input_text, temperature=0.6, top_k=3):
        self._model.eval()
        q_ids = torch.tensor(
            text_to_ids(
                self.tokenizer.engine,
                self.tokenizer.filter_text(input_text),
                self._model.context_len
            ),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_ids = self._model(q_ids, temperature=temperature, top_k=top_k)

        return self.tokenizer.engine.detokenize(
            generated_ids.squeeze(0).tolist()
        )
