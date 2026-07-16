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


while True:
    try:
        user_input = input(">>> ")
        if user_input.lower() == 'exit':
            break

        print(ask(user_input))

    except (KeyboardInterrupt, EOFError):
        break
