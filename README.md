# Ex-01---Developing-a-Neural-Network-Regression-Model
Developing a Neural Network Regression Model
# AIM
To develop a neural network regression model for the given dataset.

# THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

# Neural Network Model
Include the neural network model diagram.

# DESIGN STEPS
STEP 1: Generate Dataset
Create input values from 1 to 50 and add random noise to introduce variations in output values .

STEP 2: Initialize the Neural Network Model
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

STEP 3: Define Loss Function and Optimizer
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

STEP 4: Train the Model
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

STEP 5: Plot the Loss Curve
Track the loss function values across epochs to visualize convergence.

STEP 6: Visualize the Best-Fit Line
Plot the original dataset along with the learned linear model.

STEP 7: Make Predictions
Use the trained model to predict for a new input value .

# PROGRAM
# Name: Lathika .K
# Register Number:212224230140
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#GENERATE INPUT(X) AND OUTPUT(Y)
torch.manual_seed(71)
x = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * x + 1 + e
 
#PLOT THE ORIGINAL DATA
plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
 
#DEFINE THE LINEAR MODEL CLASS:
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)
        
#INITIALIZE THE MODEL
torch.manual_seed(59)
model = Model(1, 1)
 
#PRINT INITIAL WEIGHTS AND BIAS
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: Lathika .K")
print("Register No: 212224230140")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: 
{initial_bias:.8f}\n')

#DEFINE LOSS FUNCTION & OPTIMIZER
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
 
#TRAIN THE MODEL
epochs = 100
losses = []
 
for epoch in range(1, epochs + 1):
  optimizer.zero_grad()
  y_pred = model(x)
  loss = loss_function(y_pred, y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
    
#PRINT LOSS,WEIGHT, AND BIAS FOR EVERY EPOCH
print(f'epoch: {epoch:2} loss: {loss.item():10.8f}'
          f'weight: {model.linear.weight.item():10.8f}'
          f'bias: {model.linear.bias.item():10.8f}')
          
#PLOT LOSS CURVE
plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()
 
#FINAL WEIGHTS & BIAS
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName:LATHIKA .K ")
print("Register No:212224230140 ")
print(f"\nFinal weight: {final_weight:.8f}, Final bias: {final_bias:.8f}")

#BEST-FIT LINE CALCULATION
x1 = torch.tensor([x.min().item(), x.max().item()])
y1 = x1 * model.linear.weight.item() + model.linear.bias.item()
 
#PLOT THE ORIGINAL DATA &BEST-FIT LINE
plt.scatter(x, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-fit Line')
plt.legend()
plt.show()
 
#PREDICTION FOR X=120
x_new = torch.tensor([[120.0]])
y_new_pred = model(x_new).item()
print("\nName: Lathika .K")
print("Register No : 212224230140")
print(f"\nprediction for x = 120: {y_new_pred:.8f}")

# Initialize the Model, Loss Function, and Optimizer
Dataset Information
Include screenshot of the generated data

# OUTPUT
Training Loss Vs Iteration Plot Best Fit line plot Include your plot here
<img width="821" height="614" alt="Screenshot 2025-08-22 100942" src="https://github.com/user-attachments/assets/662af25a-8dcc-470c-b866-245f55415c00" />

<img width="595" height="133" alt="Screenshot 2025-08-22 100952" src="https://github.com/user-attachments/assets/8be6d60c-9d86-4eac-9804-6959d238da1e" />

<img width="708" height="53" alt="Screenshot 2025-08-22 101003" src="https://github.com/user-attachments/assets/edbf6f13-2644-4774-92b1-f5615e141e0d" />

<img width="825" height="598" alt="Screenshot 2025-08-22 101020" src="https://github.com/user-attachments/assets/af06f212-ff4c-4cc7-9162-a2328546c810" />

<img width="783" height="146" alt="Screenshot 2025-08-22 101028" src="https://github.com/user-attachments/assets/f5b4dfbc-239c-4915-afc6-03cfcb6d53d4" />

<img width="793" height="569" alt="Screenshot 2025-08-22 101046" src="https://github.com/user-attachments/assets/52f88bc0-19a0-45f7-ba1a-10e264b1ad90" />

<img width="695" height="139" alt="Screenshot 2025-08-22 101054" src="https://github.com/user-attachments/assets/d1818560-f16d-4f8d-9a29-58c37199239f" />

New Sample Data Prediction
Include your sample input and output here

# RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
