# preprocess_iris.py
import torch
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from originalData import iris_data

# Load into DataFrame
df = pd.DataFrame(
    iris_data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
)

# --- Step 1: Encode class labels ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df["class"])

# --- Step 2: Try different discretization strategies ---
strategies = ["uniform", "quantile", "kmeans"]

for strat in strategies:
    print("=" * 60)
    print(f" Discretization Strategy: {strat.upper()} ")
    print("=" * 60)

    # Discretize numerical features
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=strat)
    discretized_features = discretizer.fit_transform(df.drop(columns=["class"]))

    # Convert to DataFrame for readability
    X_df = pd.DataFrame(discretized_features, columns=df.columns[:-1])

    # Show bin edges for each feature
    for feature, edges in zip(df.drop(columns=["class"]).columns, discretizer.bin_edges_):
        print(f"\nFeature: {feature}")
        for i in range(len(edges) - 1):
            print(f"  Bin {i}: [{edges[i]:.2f}, {edges[i+1]:.2f})")

    # Show bin counts to check for sparsity
    for col in X_df.columns:
        counts = X_df[col].value_counts().sort_index()
        print(f"\n{col} bin counts:\n{counts}")

    # --- Step 3: Convert to PyTorch tensors ---
    X_tensor = torch.tensor(discretized_features, dtype=torch.long)
    y_tensor = torch.tensor(encoded_labels, dtype=torch.long)

    # Save tensors for later use
    torch.save({"X": X_tensor, "y": y_tensor}, f"iris_tensors_{strat}.pt")
    print(f"\nTensors saved to iris_tensors_{strat}.pt\n")
