import argparse
import pandas as pd
from visualizer import visualize_data
from pathlib import Path
import os

def main():
	# Get the project root directory (2 levels up from the script location)
	script_dir = Path(__file__).resolve().parent
	project_root = script_dir.parent.parent

	parser = argparse.ArgumentParser(description="Split dataset into training and testing sets.")
	parser.add_argument(
		"--dataset_path",
		type=str,
		required=True,
		help="Path to the dataset file (CSV format).",
	)
	parser.add_argument(
		"--train_ratio",
		type=float,
		default=0.8,
		help="Proportion of the dataset to include in the train split.",
	)
	parser.add_argument(
		"--visualize",
		action="store_true",
		help="Visualize the dataset before splitting.",
	)
	args = parser.parse_args()

	print(f"Dataset path: {args.dataset_path}")
	print(f"Train ratio: {args.train_ratio}")
	try:
		data = pd.read_csv(args.dataset_path, header=None, index_col=0)
		if data.empty:
			print("Error: The dataset is empty.")
			return
		if args.visualize:
			visualize_data(data)
		data = data.sample(frac=1.0, random_state=42)
		train_data = data.sample(frac=args.train_ratio)
		test_data = data.drop(train_data.index)
		train_data.to_csv(os.path.join(project_root,"data/processed/train_data.csv"))
		test_data.to_csv(os.path.join(project_root,"data/processed/test_data.csv"))

	except FileNotFoundError:
		print(f"Error: The file {args.dataset_path} was not found.")
		return
	except pd.errors.ParserError:
		print(f"Error: The file {args.dataset_path} is not a valid CSV.")
		return
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		return


if __name__ == "__main__":
	main()