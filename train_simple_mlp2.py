import numpy as np
import torch
import torch.nn as nn

# Custom matrix multiplication (matmul) function
def matmul(inputs, weights, device="cuda"):
    print("inputs.shape", inputs.shape)
    print("weights.shape", weights.shape)
    assert inputs.shape[-1] == weights.shape[0]
    result = torch.zeros(inputs.shape[0], weights.shape[1], device=device)
    print(result.shape)
    result = result.to(device)
    for i in range(inputs.shape[0]):
        for j in range(weights.shape[1]):
            for k in range(weights.shape[0]):
                result[i, j] += inputs[i, k] * weights[k,j]
    return result

# Custom MSE loss function
def mean_squared_error_loss(outputs, targets):
    # Flatten the targets if they are not one-hot encoded and match the shape of the outputs
    loss = torch.mean((outputs - targets) ** 2)
    return loss

# Custom Autograd function for linear layer with manual forward and backward pass
class CustomAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias=True):
        ctx.save_for_backward(inputs, weights, bias)
        print("forwaaard fn")
        print(inputs.shape)
        print(weights.shape)
        print(weights.t().shape)
        outputs = matmul(inputs, weights.t())  # matrix multiplication
        if bias is not None:
            outputs += bias  # adding bias term
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        print("grad_output.shape: ", grad_output.shape)
        print("grad_outputt: ", grad_output)
        print("weights.shape: ", weights.shape)
        print("weights: ", weights)
        if ctx.needs_input_grad[0]:
            grad_input = matmul(grad_output, weights)  # Gradient of inputs
        print("grad_output.t(): ", grad_output.t())
        print("grad_output.t().shape: ", grad_output.t().shape)
        print("inpputs: ", inputs)
        print("inputs.shape: ", inputs.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = matmul(grad_output.t(), inputs)  # Gradient of weights

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)  # Gradient of bias (sum across batch)

        return grad_input, grad_weight, grad_bias

# Custom Linear Layer
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.rand(out_features, in_features))  # Initialize weights
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features))  # Initialize bias

    def forward(self, inputs):
        return CustomAutogradFunction.apply(inputs, self.weights, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Define the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            LinearLayer(28 * 28, 512),  # Input layer to hidden layer 1
            LinearLayer(512, 512),       # Hidden layer 1 to hidden layer 2
            LinearLayer(512, 10)         # Hidden layer 2 to output layer (10 classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# Training parameters
X_train = torch.randn(1000, 28 * 28).to(device="cuda")  # 1000 samples, flattened 28x28 images
y_train = torch.randint(0, 10, (1000,)).to(device="cuda")  # 1000 target labels (10 classes)

batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Initialize the model and move to GPU
model = NeuralNetwork().to(device="cuda")
print("model: ", model)

# Training loop without DataLoader
num_samples = X_train.shape[0]
num_batches = num_samples // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        # Get batch of data
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        inputs = X_train[start_idx:end_idx]
        targets = y_train[start_idx:end_idx]

        # Forward pass
        outputs = model(inputs)

        # Convert targets to one-hot encoded vectors for MSE (if needed)
        targets_one_hot = torch.zeros(outputs.shape).to(device="cuda")
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Compute loss
        loss = mean_squared_error_loss(outputs, targets_one_hot)

        # Zero gradients from the previous step
        model.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Update parameters manually (e.g., using SGD)
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        # Optionally, print or log the loss at intervals
        if i % 100 == 0:  # Print every 100 steps
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_batches}], Loss: {loss.item():.4f}")

