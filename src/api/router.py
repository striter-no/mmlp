from aiohttp import web
from typing import Callable, Coroutine

from .api import API_rtypes, APIRequest, APISpec, is_correct_api

class APIRouter:
    def __init__(self):
        self.routes = web.RouteTableDef()

    def create_from_api_spec(
        self,
        spec: APISpec,
        handlers: dict[str,
            Callable[[APIRequest], Coroutine[None, None, tuple[str, int]]]
        ]
    ):
        for route, endpoint in spec.routes.items():
            self._add_routing(
                handlers[route], path=route, type=endpoint.rtype
            )

    def _add_routing(
        self,
        handle: Callable[[APIRequest], Coroutine[None, None, tuple[str, int]]], path: str,
        type: API_rtypes
    ):
        @self.routes.route(method=type.value, path=path)
        async def _wrapper(request: web.Request):
            try:
                j = await request.json()
            except Exception as ex:
                print(f"[main][test_handler] failed to get json: {ex}")
                return web.Response(text="Fail: invalid data encoding", status=403)

            if not is_correct_api(j):
                return web.Response(text="Fail: incorrect API formatting", status=403)

            api_r = APIRequest.parse(j)
            t, s = await handle(api_r)

            return web.Response(text=t, status=s)

    def run_server(self, host: str, port: int):
        app = web.Application()
        app.add_routes(self.routes)
        web.run_app(app, host=host, port=port)
