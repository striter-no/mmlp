from api.api import API_dtypes, API_rtypes, APIEndpoint, APIRequest
from api.router import APIRouter, APISpec

async def test_handler(r: APIRequest) -> tuple[str, int]:
    return f"Echo for {r.identity}: {r.input}", 200

def main():
    spec = APISpec()
    spec.add_endpoint(APIEndpoint(
        description="Just some test endpoint",
        input_type=API_dtypes.TEXT,
        output_type=API_dtypes.TEXT,
        possible_returns={200: "Ok"},
        name="test",
        route="/test",
        rtype=API_rtypes.GET
    ))

    router = APIRouter()
    router.create_from_api_spec(spec, {
        "/test": test_handler
    })

    router.run_server(
        host="127.0.0.1",
        port=9000
    )

if __name__ == "__main__":
    main()
