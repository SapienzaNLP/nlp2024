from pathlib import Path

from datasets import load_dataset
from tap import Tap


class ScriptArgs(Tap):
    hf_dataset_name: str = (
        "stanfordnlp/sst2"  # The name of the Hugging Face dataset to convert to JSON
    )
    output_dir: Path = "data"  # The directory to save the JSON file to

    def process_args(self):
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define train, validate and test file paths
        self.train_file = self.output_dir / "train.jsonl"
        self.validate_file = self.output_dir / "validate.jsonl"
        self.test_file = self.output_dir / "test.jsonl"


def run(args: ScriptArgs):
    dataset = load_dataset(args.hf_dataset_name)

    # Save the train, validate and test splits to JSON files
    dataset["train"].to_json(args.train_file, orient="records", lines=True)
    dataset["validation"].to_json(args.validate_file, orient="records", lines=True)
    dataset["test"].to_json(args.test_file, orient="records", lines=True)


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)
