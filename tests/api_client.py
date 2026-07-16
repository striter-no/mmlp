import asyncio
import os

from cloud.cloud_cli import MLPClient

token = os.environ["MLP_CLIENT_TOKEN"]

async def main():
    client = MLPClient(
        base_url="http://127.0.0.1:9000",
        token=token
    )

    resp = await client.request("hello world!")
    print(resp)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
