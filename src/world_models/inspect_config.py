"""Print a proposed experiment configuration without running an experiment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(config, indent=2))
    print("Configuration inspection only; training is not implemented.")


if __name__ == "__main__":
    main()
