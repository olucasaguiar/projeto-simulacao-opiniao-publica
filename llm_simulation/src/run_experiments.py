import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import subprocess

PROMPTS = [
    "natural",
    "key_value",
    "markdown",
    "json_prompt",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="llama3",
    choices=["llama3", "gemma3", "qwen3"],
)
args = parser.parse_args()
MODEL = args.model

def run(style):
    subprocess.run(
        [
            sys.executable,
            "./llm_simulation/src/get_data.py",
            "--style",
            style,
            "--model",
            MODEL
        ],
        check=True,
    )

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, PROMPTS))