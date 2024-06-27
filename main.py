import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to generate synthetic regression data
def generate_data(samples, features=20, noise=0.1, random_state=42):
    return make_regression(n_samples=samples, n_features=features, noise=noise, random_state=random_state)

# Benchmark function for sklearn models
def benchmark_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    mse = mean_squared_error(y_test, y_pred)

    return training_time, prediction_time, mse

# Benchmark function for PyTorch neural network
def benchmark_neural_network(X_train, y_train, X_test, y_test, device):
    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(20, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Convert data to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    start_time = time.time()
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time

    # Prediction
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        y_pred_torch = model(X_test_torch).cpu().numpy()
    prediction_time = time.time() - start_time

    mse = mean_squared_error(y_test, y_pred_torch)

    return training_time, prediction_time, mse

# Function to estimate total time based on smaller sample times
def estimate_total_time(sample_times, sample_sizes, total_size):
    # Linear fit to estimate time for total_size
    fit = np.polyfit(sample_sizes, sample_times, 1)
    estimated_time = np.polyval(fit, total_size)
    return estimated_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regression': SVR(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
}

sample_sizes = [1000, 2000, 4000, 8000, 10000]
final_sample_size = 100000
results = {
    'Model': [],
    'Training Time (s)': [],
    'Prediction Time (s)': [],
    'MSE': []
}

# Collect sample times for estimating total time
sample_times = {model_name: [] for model_name in models}
if device.type == 'cuda':
    sample_times['PyTorch Neural Network'] = []

for sample_size in sample_sizes:
    print(f"Testing on sample size: {sample_size}")
    X, y = generate_data(sample_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        print(f"Benchmarking {model_name} on sample size {sample_size}...")
        training_time, prediction_time, mse = benchmark_model(model, X_train, y_train, X_test, y_test)
        sample_times[model_name].append(training_time + prediction_time)

    if device.type == 'cuda':
        print(f"Benchmarking PyTorch Neural Network on sample size {sample_size}...")
        training_time, prediction_time, mse = benchmark_neural_network(X_train, y_train, X_test, y_test, device)
        sample_times['PyTorch Neural Network'].append(training_time + prediction_time)

# Estimate total time for final test size
estimated_times = {model_name: estimate_total_time(times, sample_sizes, final_sample_size)
                   for model_name, times in sample_times.items()}

print("\nEstimated times for final sample size:")
for model_name, est_time in estimated_times.items():
    print(f"{model_name}: {est_time:.4f} seconds")

# Perform the final test on the full dataset
print("\nPerforming final test on full dataset...")
X, y = generate_data(final_sample_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

total_start_time = time.time()

for model_name, model in models.items():
    print(f"Benchmarking {model_name} on full dataset...")
    training_time, prediction_time, mse = benchmark_model(model, X_train, y_train, X_test, y_test)
    results['Model'].append(model_name)
    results['Training Time (s)'].append(training_time)
    results['Prediction Time (s)'].append(prediction_time)
    results['MSE'].append(mse)

if device.type == 'cuda':
    print("Benchmarking PyTorch Neural Network on full dataset...")
    training_time, prediction_time, mse = benchmark_neural_network(X_train, y_train, X_test, y_test, device)
    results['Model'].append('PyTorch Neural Network')
    results['Training Time (s)'].append(training_time)
    results['Prediction Time (s)'].append(prediction_time)
    results['MSE'].append(mse)

total_end_time = time.time()
total_time = total_end_time - total_start_time

results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Average times
average_training_time = results_df['Training Time (s)'].mean()
average_prediction_time = results_df['Prediction Time (s)'].mean()

print(f"\nAverage Training Time: {average_training_time:.4f} seconds")
print(f"Average Prediction Time: {average_prediction_time:.4f} seconds")
print(f"\nTotal Time for all tests: {total_time:.4f} seconds")
