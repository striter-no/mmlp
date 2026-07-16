import os
import torch

from mlf.network import Network

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

    print(f"[main] trainable params: {beautify_params(nn._model.get_n_params(True))}")
    print(f"[main] all params: {beautify_params(nn._model.get_n_params())}")

    print(f"[main] loaded: context: {settings.context_len}, neurons: {settings.hidden_neurons}")
    print("[main] 'exit' for exit")

    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() == 'exit':
                break

            # print(nn.predict(user_input))
            for t in nn.predict_stream(user_input):
                print(t, end="", flush=True)
            print()

        except (KeyboardInterrupt, EOFError):
            break
