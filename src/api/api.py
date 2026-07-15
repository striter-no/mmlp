from enum import Enum

from attr import dataclass


class API_rtypes(Enum):
    POST = "post"
    GET = "get"

class API_dtypes(Enum):
    JSON = "json"
    TEXT = "text"
    BASE64 = "b64"
    URL = "url"

@dataclass
class APIEndpoint:
    name: str # short name, e.g. chat, email etc
    route: str # base_url/>route<
    rtype: API_rtypes

    input_type: API_dtypes # what accepts as input
    output_type: API_dtypes # what returns

    possible_returns: dict[int, str] # status: reason
    description: str

@dataclass
class APIRequest:
    name: str
    identity: str
    input: str

    @staticmethod
    def parse(j: dict) -> 'APIRequest':
        return APIRequest(
            identity=j["identity"],
            input=j["input"],
            name=j["name"]
        )


class APISpec:
    def __init__(self) -> None:
        self.routes: dict[str, APIEndpoint] = dict()

    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        self.routes[endpoint.route] = endpoint

def is_correct_api(j: dict) -> bool:
    if len(j.keys()) != 3:
        return False

    if j.get("name") is None:
        return False

    if j.get("identity") is None:
        return False

    if j.get("input") is None:
        return False

    return True
