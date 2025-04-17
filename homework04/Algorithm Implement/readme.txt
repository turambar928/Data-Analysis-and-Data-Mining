This repository contains a Python implementation of the Overlap K-Means (OKM) algorithm, based on the paper: Cluster Overlap as Objective Function.

Overview
Standard K-Means partitions data into non-overlapping clusters. However, in real-world data, clusters often overlap, and K-Means may not handle this gracefully.

Overlap K-Means improves upon this by incorporating an overlap score based on the ratio of intra-cluster distance to inter-cluster distance, enabling more robust clustering when cluster boundaries are ambiguous.

Features
✅ Support for global or local overlap-based clustering

✅ K-nearest-neighbor (KNN) acceleration for overlap computation

✅ Adaptive gamma threshold if not provided

✅ Compatible with NumPy and scikit-learn ecosystem

Requirements
	numpy

	scikit-learn

	matplotlib (optional, for visualization)

Install via pip:

pip install numpy scikit-learn matplotlib