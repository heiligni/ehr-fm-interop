import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistics on the data")
    parser.add_argument("data", type=str, help="Path to the data file")
    args = parser.parse_args()

    print(f"Running statistics on {args.data}")
