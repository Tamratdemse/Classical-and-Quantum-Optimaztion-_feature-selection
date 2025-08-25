import torch
from originalData import iris_data

# Convert data to numeric tensors in one step
def prepare_iris_kde_tensor(iris_data):
    # Feature/label separation with numeric encoding
    features = torch.FloatTensor([row[:-1] for row in iris_data])  # (150, 4)
    labels = torch.LongTensor([
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}[row[-1]] 
        for row in iris_data
    ])  # (150,)
    
    # Combine into single dictionary tensor
    iris_tensor = {
        'features': features,
        'labels': labels
    }
    
    # Save as single .pt file
    torch.save(iris_tensor, "iris_d_kde.pt")
    return iris_tensor

