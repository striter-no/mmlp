import os
import torch
import argparse
import string
import logging

from accelerate import Accelerator

from mlf.datastorage import DataStorage
from mlf.network import Network
from mlf.tokenizer import Tokenizer
from mlf.training import Training
from mlf.settings import ModelSettings

from mlc.datasets import Dataset


from math import floor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def beautify_params(n: int) -> str:
    K_const = 10**3
    M_const = 10**6
    B_const = 10**9

    if n > B_const:
        return f"{n / B_const:.1f}M"
    if n > M_const:
        return f"{n / M_const:.1f}M"
    if n > K_const:
        return f"{n / K_const:.1f}K"

    return f"{n}"

def beautify_time(n_sec: int) -> tuple[str, tuple[int, int, int]]:
    hours = n_sec // 3600
    minutes = n_sec / 60 - hours * 60
    sec = n_sec - hours * 3600 - floor(minutes) * 60

    return f"{hours}h{round(minutes)}m{sec}s", (hours, round(minutes), sec)

def train_network(nn, storage, tokenizer, epochs, stop_error, batch_size, cost_rate, start_from):
    trainer = Training(model=nn, storage=storage, tokenizer=tokenizer)
    acc = nn.accelerator

    if acc.is_main_process:
        logger.info("[main] making training data")
    trainer.prepare_data(epochs=epochs, batch_size=batch_size, start_epoch=start_from)
    trainer.show_info()

    if acc.is_main_process:
        logger.info(f"[main] starting to train NN for {epochs} epochs")
    for i in range(epochs):
        info = trainer.train_epoch()
        m_ep = info.time_elapsed / 60
        eta_sec = int(m_ep * 60 * (epochs - i - 1))

        ptime, (hours, mins, sec) = beautify_time(eta_sec)

        cost_str = ""
        if cost_rate > 0:
            total_cost = hours * cost_rate + mins * cost_rate / 60 + sec * cost_rate / 3600
            cost_str = f" | Cost: ${total_cost:.2f}"

        if acc.is_main_process:
            logger.info(f"- epoch {i+1}/{epochs} (trained in {m_ep:.3f}m), error: {info.error:.4f} | ETA: {ptime}{cost_str}")

        if info.error < stop_error:
            if acc.is_main_process:
                logger.info("[main] stopped training due the error being trained")
            break

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Run without interactive tests")
    parser.add_argument("--continue-learn", action="store_true", help="Load model and continue training")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cudnn")
    parser.add_argument("--epochs", type=int, help="How many epochs to train", default=60)
    parser.add_argument("--epochs-start", type=int, help="From what epoch to start", default=0)
    parser.add_argument("--batch", type=int, help="Number of batch requests", default=64)
    parser.add_argument("--pairs", type=int, help="Number of Q/A pairs to load", default=2000)
    parser.add_argument("--tokens", type=int, help="Vocabulary size", default=3000)
    parser.add_argument("--context", type=int, help="Context length (tokens)", default=200)
    parser.add_argument("--embed", type=int, help="Embedding dimensions", default=128)
    parser.add_argument("--neurons", type=int, help="Hidden neurons in GRU", default=512)
    parser.add_argument("--temp", type=float, help="Default temperature for inference", default=0.8)
    parser.add_argument("--cost", type=float, help="How much costs 1 hour of training (-1 for disabling this info)", default=-1)
    parser.add_argument("--top-k", type=int, help="Default top-k for inference", default=5)
    parser.add_argument("--stop-error", type=float, help="When to stop training", default=1.3)
    parser.add_argument("--dropout", type=float, help="Dropout rate", default=0.3)
    parser.add_argument("--layers", type=int, help="Number of GRU layers in encoder", default=2)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = not args.disable_cudnn

    settings = ModelSettings(
        dataset_path="knkarthick/dialogsum",
        dataset_name="default",
        column_text="dialogue",
        column_title="summary",
        pairs_num=args.pairs,
        tokens_num=args.tokens,
        encoder_layers=args.layers,
        context_len=args.context,
        embedding_dims=args.embed,
        hidden_neurons=args.neurons,
        default_temp=args.temp,
        default_top_k=args.top_k,
        alphabet=string.printable,
        dropout=args.dropout
    )

    logger.info(f"[main] loading dataset '{settings.dataset_name}'")
    dial_ds = Dataset(column_text=settings.column_text, column_title=settings.column_title)
    dial_ds.load(settings.dataset_path, settings.dataset_name)

    storage = DataStorage(cache_path=".cache", dataset=dial_ds)
    storage.load_dialogues(settings.pairs_num)

    logger.info("[main] making tokens")
    tokenizer = Tokenizer(
        cache_path=".cache",
        storage=storage
    )
    tokenizer.load_tokens(settings.tokens_num)

    nn = Network(tokenizer=tokenizer, settings=settings)
    if args.continue_learn:
        logger.info(f"[main] loaded model, continuing to learn from {args.epochs_start} epoch")
        nn.raw_model.load("./.cache/nn.pth", nn.device)

    if nn.accelerator.is_main_process:
        logger.info(f"[main] trainable params: {beautify_params(nn.raw_model.get_n_params(True))}")
        logger.info(f"[main] all params: {beautify_params(nn.raw_model.get_n_params())}")

    try:
        train_network(nn, storage, tokenizer, args.epochs, args.stop_error, args.batch, args.cost, args.epochs_start)
    except (KeyboardInterrupt, EOFError):
        logger.info("[main] interrupted")
    except Exception as ex:
        logger.info(f"[main] during training exception occured: {ex}")

    if nn.accelerator.is_main_process:
        logger.info("[main] done, saving results")
        nn.save_to_file("./.cache/nn.pth")
        settings.save_to_file("./.cache/settings.json")

        logger.info("[main] done")
