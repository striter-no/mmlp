from api.rclient import APIClient

from .cloud_spec import CLOUD_BASIC_SPEC

class MLPClient:
    def __init__(self, token: str, base_url: str):
        self.apicli = APIClient(
            base_url=base_url,
            identity=token,
            spec=CLOUD_BASIC_SPEC
        )

    async def request(self, text: str) -> str:
        r, s = await self.apicli.api_request("request", text)

        if s != 200:
            raise RuntimeError(f"Failed to make API request: {s} <- {r}")

        return r["completion"] if isinstance(r, dict) else ""

    async def close(self):
        await self.apicli.end()
