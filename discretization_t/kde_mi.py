import torch, math

# -----------------------
# 1) Load preprocessed Iris tensor
# -----------------------
iris_tensor = torch.load("iris_d_kde.pt")
X = iris_tensor['features']   # shape (150, 4)
y = iris_tensor['labels']     # shape (150,)

# -----------------------
# 2) Gaussian KDE functions (with leave-one-out)
# -----------------------
def gaussian_kernel(u):
    return (1.0 / math.sqrt(2*math.pi)) * torch.exp(-0.5 * u**2)

def kde_density_1d(x, data, h):
    """LOO 1D KDE: density of scalar x w.r.t. dataset (excluding x itself)."""
    if data.numel() == 0:
        return torch.tensor(1e-12)  # avoid div-by-zero when no class samples
    diffs = (x - data) / h
    vals = gaussian_kernel(diffs)
    return vals.mean() / h

# -----------------------
# 3) Mutual Information (feature vs discrete class)
# -----------------------
def mutual_information_feature_class(feature, labels, h):
    """
    Estimate I(X;Y) between one continuous feature and discrete class Y.
    """
    n = feature.shape[0]
    mi_vals = []

    for i in range(n):
        xi, yi = feature[i], labels[i]

        # Exclude self for LOO
        mask = torch.arange(n) != i
        data_x = feature[mask]
        data_y = labels[mask]

        # KDE p(x_i | y_i)
        same_class = data_x[data_y == yi]
        px_given_y = kde_density_1d(xi, same_class, h)

        # KDE p(x_i)
        px = kde_density_1d(xi, data_x, h)

        # Prior p(y)
        py = (labels == yi).float().mean()

        # MI contribution: log(p(x|y) / p(x))
        if px > 0 and px_given_y > 0:
            mi_vals.append(torch.log(px_given_y / px))

    return torch.stack(mi_vals).mean().item()

# -----------------------
# 4) Bandwidth rule-of-thumb (Silverman)
# -----------------------
def silverman_bandwidth(data):
    n = len(data)
    std = data.std(unbiased=True).item()
    return 1.06 * std * (n ** (-1/5))

# -----------------------
# 5) Run MI calculation for all Iris features
# -----------------------
feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

for f in range(X.shape[1]):
    feat = X[:, f]
    h = silverman_bandwidth(feat)
    mi = mutual_information_feature_class(feat, y, h)
    print(f"{feature_names[f]:<15} : {mi:.4f} nats ({mi/math.log(2):.4f} bits)")
