import matplotlib.pyplot as plt

def visualize_data(data):
	"""
	Visualizes the dataset with histograms for each feature, color-coding by class.

	Parameters:
	data (DataFrame): A pandas DataFrame containing the dataset.

	Returns:
	None
	"""
	if len(data.columns) < 2:
		print("Error: Data must have at least two columns")
		return

	class_column = data.iloc[:, 1]

	plt.figure(figsize=(15, 10))
	plt.suptitle('Distribution of the Features', fontsize=16, y=0.98)

	graph_size = int((len(data.columns) - 2) ** 0.5) + 1
	for i in range(2, len(data.columns) - 1):
		plt.subplot(graph_size - 1, graph_size, i - 1)
		for label, color in zip(['B', 'M'], ['blue', 'red']):
			mask = class_column == label
			plt.hist(data.iloc[:, i][mask], alpha=0.5, color=color, label=label, bins=20)

	plt.legend()
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()