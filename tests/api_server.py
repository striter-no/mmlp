import os
import logging
import torch
from mlf.network import Network
from cloud.cloud_nn import CloudModelsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    os.makedirs(".cache", exist_ok=True)

    logger.info("[server] loading model")

    nn, settings = Network.load_from_file(
        model_path="./.cache/nn.pth",
        settings_path="./.cache/settings.json"
    )
    torch.backends.cudnn.enabled = False

    logger.info("[server] starting cmm")

    manager = CloudModelsManager(nn)
    manager.run_server(port=9000)
