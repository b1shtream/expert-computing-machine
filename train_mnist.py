import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        outputs = torch.matmul(inputs, weights.t())  # matrix multiplication
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
            grad_input = torch.matmul(grad_output, weights)  # Gradient of inputs
        print("grad_output.t(): ", grad_output.t())
        print("grad_output.t().shape: ", grad_output.t().shape)
        print("inpputs: ", inputs)
        print("inputs.shape: ", inputs.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), inputs)  # Gradient of weights

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

# Loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to the range [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
print("training started...........")
# Define the optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, targets) in enumerate(train_loader):
        # Move data to the correct device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Convert targets to one-hot encoded vectors for MSE
        targets_one_hot = torch.zeros(outputs.shape).to(device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Compute loss
        loss = mean_squared_error_loss(outputs, targets_one_hot)

        # Zero gradients from the previous step
        model.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Update parameters manually using SGD
        optimizer.step()

        # Optionally, print or log the loss at intervals
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print every 100 steps
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

