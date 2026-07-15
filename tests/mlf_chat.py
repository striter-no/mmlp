import os
import torch

from mlf.network import Network

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    os.makedirs(".cache", exist_ok=True)

    nn, settings = Network.load_from_file(
        model_path="./.cache/nn.pth",
        settings_path="./.cache/settings.json"
    )

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
