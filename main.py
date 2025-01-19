import torch
from torch import nn
import torch.optim.adam
import torch.optim.sgd
import matplotlib.pyplot as plt
import torch.optim.sgd
class objective_fn(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(1,5) #remove false bias
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(5,1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.tensor:
        x = torch.square(self.w1(x))
        return self.w2(x)

def random_from_a_b(size, a,b):
    return torch.rand(size,1)*(b-a) + a

# x = torch.rand(1000, 1)
x = random_from_a_b(2000,-1.5, 3.5)
# print(x.min(),x.max())
# print(x[:10])
y = 2*((x)**2) - 4*x - 6
# print(x.size(), y.size())
tt_split = int(0.8*len(x))
x_train,y_train = x[tt_split:], y[tt_split:]
x_test,y_test = x[:tt_split],y[:tt_split]


model = objective_fn()
optimizer_fn = torch.optim.SGD(model.parameters(), lr=0.001)
# loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
epochs = 500

for epoch in range(epochs):
    model.train()

    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer_fn.zero_grad()
    loss.backward()
    optimizer_fn.step()

    # if(epoch+1)%100 == 0:
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        # print(loss.item(), loss) 
model.eval()
with torch.inference_mode():
    y_test_pred = model(x_test)
    print("loss of objective: ", loss.item(), loss)



plt.figure(figsize=(10, 5))
# Plot training data
plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', label='Training Data')

# Plot test data
plt.scatter(x_test.numpy(), y_test.numpy(), color='green', label='Test Data')

# Plot predicted values
plt.scatter(x_test.numpy(), y_test_pred.numpy(), color='red', label='Predicted Values')

plt.title('Quadratic Function Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.show()
#--------------------------------------------------
class NOM(nn.Module):
    def __init__(self, new_layer, old_model):
        super().__init__()
        self.new_layer = new_layer
        self.old_model = old_model

    def forward(self, x):
        x = self.new_layer(x)
        # x = x.view(-1, 1)  
        x = self.old_model(x)
        return x
    
new_layer = nn.Linear(1,1)
new_model = NOM(new_layer, model)

for params in new_model.old_model.parameters():
    params.requires_grad = False

# combined_net = NOM(new_layer, model)

optimizer = torch.optim.Adam(new_model.new_layer.parameters(), lr=0.01)
loss_fn = torch.nn.L1Loss()
input_value = torch.tensor([2.0], requires_grad=True)  # Starting point

for i in range(100):
    new_model.train()
    optimizer.zero_grad()
    output = new_model(input_value)
    # loss = torch.mean(output)  # Minimize the output of the fixed network
    loss = loss_fn(output, torch.zeros_like(output))
    loss.backward()
    optimizer.step()
    # if i%10 == 0:
        # print("loss: ",loss)

# print(output)
# result = torch.matmul(input_value, new_model.new_layer.weight.T)
# print(result)

# print(new_model(input_value))
new_model.eval()
with torch.inference_mode():
    print("loss: ",loss.item())
    print("output", new_model(input_value))
    print("first layer into obj_fn: ", new_model.new_layer.weight.item(), new_model.new_layer.bias.item())
    print(input_value)
    # print(new_model.old_model(torch.matmul(input_value, new_model.new_layer.weight.T)))
    # print(list(new_model.parameters()))
    for x in list(new_model.parameters()):
        print(x.size()) 
