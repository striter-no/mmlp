import os
from mlf.network import Network
from cloud.cloud_nn import CloudModelsManager
import torch

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)

    print("[server] loading model")
    nn, settings = Network.load_from_file(
        model_path="./.cache/nn.pth",
        settings_path="./.cache/settings.json"
    )

    torch.backends.cudnn.enabled = False

    print("[server] starting cmm")
    manager = CloudModelsManager(nn)
    manager.run_server(port=9000)
