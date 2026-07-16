from enum import Enum
from attr import dataclass

class API_rtypes(Enum):
    POST = "POST"
    GET = "GET"

class API_dtypes(Enum):
    JSON = "json"
    TEXT = "text"
    BASE64 = "b64"
    URL = "url"

@dataclass
class APIEndpoint:
    name: str
    route: str
    rtype: API_rtypes
    input_type: API_dtypes
    output_type: API_dtypes
    possible_returns: dict[int, str]
    description: str

@dataclass
class APIRequest:
    identity: str
    input: str

    @staticmethod
    def parse(j: dict) -> 'APIRequest':
        return APIRequest(
            identity=j.get("identity", ""),
            input=j.get("input", ""),
        )

class APISpec:
    def __init__(self) -> None:
        self.routes: dict[str, APIEndpoint] = dict()

    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        self.routes[endpoint.route] = endpoint

def is_correct_api(j: dict) -> bool:
    if not isinstance(j, dict):
        return False

    if "identity" not in j or "input" not in j:
        return False
    return True
