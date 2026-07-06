import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import subprocess

FEATURES = [
   "full", "demographics_only", "no_religion", "no_income", "no_memory", "no_interest", "politics_only"
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="llama3",
    choices=["llama3", "gemma3", "qwen3"],
)
args = parser.parse_args()
MODEL = args.model

def run(feature_style):
    subprocess.run(
        [
            sys.executable,
            "./llm_simulation/src/get_data.py",
            "--feature_style",
            feature_style,
            "--model",
            MODEL
        ],
        check=True,
    )

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=7) as executor:
        list(executor.map(run, FEATURES))