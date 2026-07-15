import os
from mlf.network import Network

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)

    print("[main] Loading model (this will auto-init dataset and tokenizer)...")
    nn, settings = Network.load_from_file(
        model_path="./.cache/nn.pth",
        settings_path="./.cache/settings.json"
    )

    print(f"[main] Loaded! Context: {settings.context_len}, Neurons: {settings.hidden_neurons}")
    print("[main] 'exit' for exit")

    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() == 'exit':
                break
            # predict использует default_temp и default_top_k из настроек
            print(nn.predict(user_input))
        except (KeyboardInterrupt, EOFError):
            break
