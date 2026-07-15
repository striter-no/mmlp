import json
from types import TracebackType
from urllib.parse import urljoin

import aiohttp

from .api import API_dtypes, APIEndpoint, APISpec

class APIClient:
    def __init__(self, base_url: str, identity: str, spec: APISpec) -> None:
        self.base_url: str = base_url
        self.spec: APISpec = spec
        self.identity: str = identity
        self.qtable: dict[str, APIEndpoint] = {
            endp.name: endp for _, endp in spec.routes.items()
        }

        self.session = aiohttp.ClientSession()

    async def api_request(self, name: str, input: str | dict) -> tuple[str | dict, int]:
        if name not in self.qtable:
            raise RuntimeError("No such method in API spec: " + name)

        endp = self.qtable[name]
        r = {
            "input": (
                input if endp.input_type != API_dtypes.JSON else
                json.dumps(input)
            ),
            "identity": self.identity,
            "name": name,
        }

        async with self.session.request(
            endp.rtype.value,
            url = urljoin(self.base_url, endp.route),
            json = r
        ) as resp:
            if resp.status != 200:
                return await resp.text(), resp.status

            match endp.output_type:
                case API_dtypes.JSON:
                    return await resp.json(), resp.status
                case API_dtypes.TEXT:
                    return await resp.text(), resp.status

        raise RuntimeError("Failed, Unknown reason")

    async def end(self):
        await self.session.close()

    async def __aenter__(self) -> "APIClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.end()
