import argparse
import pandas as pd


def main():
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
	args = parser.parse_args()

	print(f"Dataset path: {args.dataset_path}")
	print(f"Train ratio: {args.train_ratio}")
	# Load the dataset
	try:
		data = pd.read_csv(args.dataset_path)
		print(data)
	except FileNotFoundError:
		print(f"Error: The file {args.dataset_path} was not found.")
		return
	except pd.errors:
		print(f"Error: The file {args.dataset_path} is not a valid CSV.")
		return
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		return
	if data.empty:
		print("Error: The dataset is empty.")
		return


if __name__ == "__main__":
	main()