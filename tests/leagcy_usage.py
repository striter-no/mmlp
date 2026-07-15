import os
import string
import time
import argparse

from torch import nn
import torch
import torch.optim as optim

import mlc.vocab as vocab
import mlc.datasets as ds

from torch.utils.data import TensorDataset, DataLoader
from mlc.tokenizer import TokenEngine
from mlc.network import AutoregNavigator, text_to_ids


def load_pairs() -> tuple[list[tuple[str, str]], list[str]]:
    os.makedirs(".cache", exist_ok=True)

    dial_ds = ds.Dataset(column_text="dialogue", column_title="summary")
    dial_ds.load("knkarthick/dialogsum", "default")

    ds_texts_raw = [dial_ds.get_text(i).lower() for i in range(min(10_000, dial_ds.max_texts()))]
    ds_texts = []
    ds_pairs = []

    for text in ds_texts_raw:
        if text is None:
            continue

        pair = []
        for replic in text.split("\n"):
            cleared = " ".join(replic.split()[1:])
            if len(pair) == 2:
                ds_pairs.append(pair.copy())
                pair = []

            pair.append(cleared)
            ds_texts.append(cleared)

    return ds_pairs, ds_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--context", type=int, help="How many tokens can model have as an input and output", default=30)
    parser.add_argument("--epochs", type=int, help="How many epochs will model train", default=1_000)
    parser.add_argument("--path", type=str, help="Path where to save/load model weights", default="./nn.pth")
    parser.add_argument("--device", type=str, help="Prefered device to use", default="auto")
    parser.add_argument("--pairs", type=int, help="Number of pairs to choose for training, -1 for all of them", default=150)
    parser.add_argument("--tokens", type=int, help="Number of tokens in model's vocabulary", default=1000)
    parser.add_argument("--load", action="store_true", help="Load this model, without training")
    parser.add_argument("--learn", action="store_true", help="Load this model, continue training")
    parser.add_argument("--neurons", type=int, help="Number of the neurons in the model", default=512)

    args = parser.parse_args()

    context_max_tokens = args.context
    max_dimensions = context_max_tokens
    epochs = args.epochs
    model_path = args.path
    alphabet = string.printable# + "йцукенгшщзхъфывапролджэячсмитьбю"

    torch.set_num_threads(os.cpu_count() or 8)

    device = args.device

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"[main] using device: {device}")

    print("[main] loading dataset")
    ds_pairs, ds_texts = load_pairs()
    if args.pairs > -1:
        ds_pairs = ds_pairs[:min(len(ds_pairs), args.pairs)]

    print(f" - got {len(ds_texts)} texts\n - got {len(ds_pairs)} answer/requests")

    print("[main] loading vocab")
    ds_vocab = vocab.get_vocab(ds_texts, alphabet=alphabet, top_n=args.tokens)
    if ds_vocab is None:
        raise RuntimeError("Failed to load vocab")

    engine = TokenEngine(ds_vocab)

    using = args.load
    learn = args.learn

    model = AutoregNavigator(
        embed_dim=128,
        context_len=context_max_tokens,
        vocab_size=engine.base,
        hidden=args.neurons,
        pad_id=engine.pad_id,
        use_attention=True,
    ).to(device)

    if learn:
        print(f"[main] continuing to learn from {model_path}")
        model.load(model_path, device)

    if not using:
        print("[main] making train data")
        train_data = []
        for q_text, a_text in ds_pairs:
            q_ids = torch.tensor(
                text_to_ids(engine, "".join([c for c in q_text if c in alphabet]), context_max_tokens),
                dtype=torch.long
            )

            a_ids = text_to_ids(engine, a_text, context_max_tokens)
            train_data.append((q_ids, torch.tensor(a_ids, dtype=torch.long)))

        X = torch.stack([q for q, _ in train_data])
        Y = torch.stack([a for _, a in train_data])  # (N, context_max_tokens), long

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train_start = time.time()
        last_epoch = 0
        print(f"[main] NN training, train data: {len(train_data)} entries, {epochs} epochs")
        try:
            for epoch in range(epochs):
                start = time.time()
                total_loss = 0

                for batch_X, batch_Y in dataloader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    optimizer.zero_grad()

                    pred_logits = model(batch_X, batch_Y)
                    loss = loss_fn(pred_logits.transpose(1, 2), batch_Y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()

                if epochs > 500 and epoch % 10 != 0:
                    continue

                last_epoch = epoch
                got = time.time() - start
                print(
                    f"\r[epoch {epoch}, error: {total_loss / len(dataloader):.6f}, got here in {got/60:.2f}m] {(time.time() - train_start) / 60:.2f}m elapsed, ~{got * (epochs - epoch) / 60:.2f}m ETA",
                    end=""
                )
        except (KeyboardInterrupt, EOFError):
            print(f"\n[training interrupted at {last_epoch} epochs]")

        print(f"\n\n[main] saving to the {model_path}")
        model.save(model_path)

    if using:
        print(f"\n[main] loading from the {model_path}")
        model.load(model_path, device)

    model.eval()

    def predict(text, temperature=0.6, top_k=3):
        q_ids = torch.tensor(
            text_to_ids(engine, "".join([c for c in text if c in alphabet]), context_max_tokens),
            dtype=torch.long
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(q_ids, temperature=temperature, top_k=top_k)

        return engine.detokenize(logits.squeeze(0).argmax(-1).tolist())

    print("\n[main] test of the NN")
    for i, (q_text, a_text) in enumerate(ds_pairs[:5] if not using else ds_pairs[:20]):
        print(f"[{i + 1}/{len(ds_pairs)}]")
        print(f"Q: {q_text}")
        print(f"A: {a_text}")
        print(f"\nNN: {predict(q_text)}")
        print()

    if not using:
        qtext = "hello"
        print(f"[loop] testing... initial: {qtext}")
        for i in range(10):
            qtext = predict(qtext)
            print(f"[{i + 1} it]: {qtext}")

    if using:
        print("\n[human] testing, \"exit\" or Ctrl+C to exit")
        while True:
            try:
                i = input(">>> ")
                if i == "exit":
                    break
                print(predict(i))
            except (KeyboardInterrupt, EOFError):
                break
