import argparse
import os
import string

from mlf.datastorage import DataStorage
from mlf.network import Network
from mlf.tokenizer import Tokenizer
from mlf.training import Training

from mlc.datasets import Dataset

def train_network():
    trainer = Training(
        model=nn, storage=storage, tokenizer=tokenizer
    )

    print("[main] making training data")
    trainer.prepare_data()
    trainer.show_info()

    print("[main] starting to train NN")
    epochs = 50
    for i in range(epochs):
        info = trainer.train_epoch()
        m_ep = info.time_elapsed / 60
        eta = m_ep * (epochs - i)

        print(f"- epoch {i+1}/{epochs} (trained in {m_ep:.3f}m), error: {info.error:.4f} | ETA: {eta:.3f}m")

        if info.error < 1.3:
            print("[main] stopped training due the error being trained")
            break

    print("[main] done, starting tests")
    nn.save_to_file("./.cache/nn.pth")

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without interactive tests")
    args = parser.parse_args()

    print("[main] loading dataset")
    dial_ds = Dataset(column_text="dialogue", column_title="summary")
    dial_ds.load("knkarthick/dialogsum", "default")
    storage = DataStorage(
        cache_path=".cache", dataset=dial_ds
    )
    storage.load_pairs(2000)

    print("[main] making tokens")
    tokenizer = Tokenizer(
        alphabet=string.printable,
        cache_path=".cache",
        storage=storage
    )
    tokenizer.load_tokens(3000)

    nn = Network(
        tokenizer, context_len=30, embedding_dims=64, hidden_neurons=256
    )

    train_network()

    if not args.headless:
        print("\n--- [Dataset Test] ---")
        for q, a in storage.pairs[:5]:
            print(f"Q: {q}")
            print(f"A: {a}")
            print(f"\nNN: {nn.predict(q)}\n")

        print("\n--- [Human Test] ---")
        test_phrases = [
            "hello, how are you?",
            "what is your name?",
            "i want to buy a car",
            "do you like pizza?"
        ]

        for phrase in test_phrases:
            print(f"Q: {phrase}")
            print(f"NN: {nn.predict(phrase)}\n")

        print("--- [Interactive] (type 'exit' to quit) ---")
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() == 'exit':
                    break
                print(f"NN: {nn.predict(user_input)}\n")
            except (KeyboardInterrupt, EOFError):
                break
