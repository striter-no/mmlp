import argparse
import subprocess
import os

# Твои настройки сервера
SERVER_IP = os.environ["MLP_MLOPS_SERVER_IP"]
SERVER_USER = os.environ.get("MLP_MLOPS_SERVER_USER", "root")
SERVER_PATH = os.environ.get("MLP_MLOPS_SERVER_PATH", "/root/mmlp")
LOCAL_MODEL_PATH = os.environ.get("MLP_MODEL_PATH", "./.cache/nn.pth")

def run_ssh(cmd: str, background: bool = False):
    if background:
        full_cmd = f"tmux new-session -d -s train 'cd {SERVER_PATH} && {cmd} 2>&1 | tee train.log'"
    else:
        full_cmd = cmd

    print(f"[ssh] {full_cmd}")
    subprocess.run(["ssh", f"{SERVER_USER}@{SERVER_IP}", full_cmd], check=False)

def update_code():
    print("[cloud] Updating code on server via Git...")
    run_ssh(f"cd {SERVER_PATH} && git pull")

def start_train(args: str):
    print("[cloud] Starting training on server...")

    cmd = f"source .venv/bin/activate && python ./tests/mlf_usage.py {args}"
    run_ssh(cmd, background=True)
    print("[cloud] Training started in background.")
    print(f"[cloud] To view logs: ssh {SERVER_USER}@{SERVER_IP} 'tail -f {SERVER_PATH}/train.log'")
    print(f"[cloud] To attach to process: ssh {SERVER_USER}@{SERVER_IP} -t 'tmux attach -t train'")

def fetch_model():
    print("[cloud] Downloading model from server...")
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    remote_path = f"{SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/.cache/nn.pth"

    subprocess.run(["scp", remote_path, LOCAL_MODEL_PATH], check=True)
    print(f"[cloud] Model saved to {LOCAL_MODEL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_update = subparsers.add_parser("update", help="Pull latest code on server")

    sp_train = subparsers.add_parser("train", help="Start training on server")
    sp_train.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the training script")

    sp_fetch = subparsers.add_parser("fetch", help="Download nn.pth from server")

    args = parser.parse_args()

    if args.command == "update":
        update_code()
    elif args.command == "train":
        start_train(" ".join(args.args) if args.args else "")
    elif args.command == "fetch":
        fetch_model()
