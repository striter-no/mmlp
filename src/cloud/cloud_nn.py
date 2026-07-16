from api.router import APIRouter
from api.api import APIRequest
from mlf.network import Network
from .cloud_spec import CLOUD_BASIC_SPEC

import string

import logging
logger = logging.getLogger(__name__)

class CloudModelsManager:
    def __init__(self, nn: Network):
        self.nn = nn
        self.router = APIRouter()

        self.router.create_from_api_spec(
            spec=CLOUD_BASIC_SPEC,
            handlers={"/request": self._handle_request}
        )

    async def _handle_request(self, api_r: APIRequest):
        text = api_r.input

        sanitized = "".join([c for c in text[:100] if c in string.printable])
        logger.info(
            f"[handler] incoming API request from {api_r.identity[:20]}, body: {sanitized}"
        )

        try:
            result = self.nn.predict(text)
            return {"completion": result}, 200
        except Exception as e:
            return {"error": str(e)}, 500

    def run_server(self, host: str = "0.0.0.0", port: int = 8080):
        self.router.run_server(host, port)
