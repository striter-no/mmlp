import requests
import os

token = os.environ["MLP_CLIENT_TOKEN"]

def ask(text: str) -> str:
    resp = requests.post("http://2.26.22.96:6000/request", json={
        "identity": token,
        "input": text
    })

    resp.raise_for_status()
    return resp.json()["completion"]


user_input = "hello"
while True:
    try:
        ans = ask(user_input)
        print(ans)

        user_input = ans
    except (KeyboardInterrupt, EOFError):
        break
