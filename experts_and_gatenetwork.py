import torch
import torch.nn as nn
import torch.optim as optim

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_experts)  
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  

# 超参数
input_dim = 10      
output_dim = 1     
num_experts = 3     
batch_size = 32     

experts = [Expert(input_dim, output_dim) for _ in range(num_experts)]

gating_network = GatingNetwork(input_dim, num_experts)

X_input = torch.randn(batch_size, input_dim)  
y_true = torch.randn(batch_size, output_dim)  


criterion = nn.MSELoss()
optimizer = optim.Adam(list(gating_network.parameters()) + [param for expert in experts for param in expert.parameters()])


def train_step():
    optimizer.zero_grad()

    gating_weights = gating_network(X_input)  

    expert_outputs = [expert(X_input) for expert in experts]  # 每个专家的输出形状是 (batch_size, output_dim)

    weighted_expert_outputs = sum(gating_weights[:, i:i+1] * expert_outputs[i] for i in range(num_experts))


    loss = criterion(weighted_expert_outputs, y_true)
    loss.backward()  
    optimizer.step() 

    return loss.item()


num_epochs = 100
for epoch in range(num_epochs):
    loss = train_step()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

