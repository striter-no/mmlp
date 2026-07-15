import asyncio

from api.api import API_dtypes, API_rtypes, APIEndpoint
from api.rclient import APIClient, APISpec

async def main():
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

    async with APIClient(
        base_url="http://127.0.0.1:9000",
        identity="1234abcdEFGH",
        spec=spec
    ) as cli:

        r, s = await cli.api_request("test", "Hello world!")
        if s != 200:
            print("Something went wrong: ", r, s)
        else:
            print("Got answer: ", r)

if __name__ == "__main__":
    asyncio.run(main())
