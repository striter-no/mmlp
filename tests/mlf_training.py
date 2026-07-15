import argparse
import os
import string

from mlf.datastorage import DataStorage
from mlf.network import Network
from mlf.tokenizer import Tokenizer
from mlf.training import Training
from mlf.settings import ModelSettings

from mlc.datasets import Dataset

def train_network(nn, storage, tokenizer, epochs, stop_error):
    trainer = Training(model=nn, storage=storage, tokenizer=tokenizer)

    print("[main] making training data")
    trainer.prepare_data()
    trainer.show_info()

    print(f"[main] starting to train NN for {epochs} epochs")
    for i in range(epochs):
        info = trainer.train_epoch()
        m_ep = info.time_elapsed / 60
        eta = m_ep * (epochs - i)

        print(f"- epoch {i+1}/{epochs} (trained in {m_ep:.3f}m), error: {info.error:.4f} | ETA: {eta:.3f}m")

        if info.error < stop_error:
            print("[main] stopped training due the error being trained")
            break

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Run without interactive tests")
    parser.add_argument("--epochs", type=int, help="How many epochs to train", default=60)
    parser.add_argument("--pairs", type=int, help="Number of Q/A pairs to load", default=2000)
    parser.add_argument("--tokens", type=int, help="Vocabulary size", default=3000)
    parser.add_argument("--context", type=int, help="Context length (tokens)", default=200)
    parser.add_argument("--embed", type=int, help="Embedding dimensions", default=128)
    parser.add_argument("--neurons", type=int, help="Hidden neurons in GRU", default=512)
    parser.add_argument("--temp", type=float, help="Default temperature for inference", default=0.8)
    parser.add_argument("--top-k", type=int, help="Default top-k for inference", default=5)
    parser.add_argument("--stop-error", type=float, help="When to stop training", default=1.3)
    parser.add_argument("--dropout", type=float, help="Dropout rate", default=0.3)
    args = parser.parse_args()

    settings = ModelSettings(
        dataset_path="knkarthick/dialogsum",
        dataset_name="default",
        column_text="dialogue",
        column_title="summary",
        pairs_num=args.pairs,
        tokens_num=args.tokens,
        context_len=args.context,
        embedding_dims=args.embed,
        hidden_neurons=args.neurons,
        default_temp=args.temp,
        default_top_k=args.top_k,
        alphabet=string.printable,
        dropout=args.dropout
    )

    print(f"[main] loading dataset '{settings.dataset_name}'")
    dial_ds = Dataset(column_text=settings.column_text, column_title=settings.column_title)
    dial_ds.load(settings.dataset_path, settings.dataset_name)

    storage = DataStorage(cache_path=".cache", dataset=dial_ds)
    storage.load_pairs(settings.pairs_num)

    print("[main] making tokens")
    tokenizer = Tokenizer(
        alphabet=settings.alphabet,
        cache_path=".cache",
        storage=storage
    )
    tokenizer.load_tokens(settings.tokens_num)

    nn = Network(tokenizer=tokenizer, settings=settings)

    train_network(nn, storage, tokenizer, args.epochs, args.stop_error)

    print("[main] done, saving results")
    nn.save_to_file("./.cache/nn.pth")
    settings.save_to_file("./.cache/settings.json")

    if not args.headless:
        print("\n--- [Dataset Test] ---")
        for q, a in storage.pairs[:5]:
            print(f"Q: {q}")
            print(f"A: {a}")
            print(f"\nNN: {nn.predict(q)}\n")

        print("\n--- [Interactive] (type 'exit' to quit) ---")
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() == 'exit':
                    break
                print(f"NN: {nn.predict(user_input)}\n")
            except (KeyboardInterrupt, EOFError):
                break
