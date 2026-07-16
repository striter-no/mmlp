import os
import logging
import torch
from mlf.network import Network

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

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    os.makedirs(".cache", exist_ok=True)

    nn, settings = Network.load_from_file(
        model_path="./.cache/nn.pth",
        settings_path="./.cache/settings.json"
    )

    logger.info(f"[main] trainable params: {beautify_params(nn.raw_model.get_n_params(True))}")
    logger.info(f"[main] all params: {beautify_params(nn.raw_model.get_n_params())}")
    logger.info(f"[main] loaded: context: {settings.context_len}, neurons: {settings.hidden_neurons}")
    logger.info("[main] 'exit' for exit")

    history = []

    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() == 'exit':
                break

            response, used_ctx, max_ctx = nn.predict(user_input, history=history)
            print(response)

            fill_percent = (used_ctx / max_ctx) * 100

            bar_length = 20
            filled_length = int(bar_length * used_ctx // max_ctx)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)

            print(f"\n[{used_ctx}/{max_ctx} tok] [{bar}] {fill_percent:.1f}%\n")

            history.append(user_input)
            history.append(response)

            if used_ctx >= max_ctx:
                logger.warning("Context is full! Clearing history...")
                history = []

        except (KeyboardInterrupt, EOFError):
            break
