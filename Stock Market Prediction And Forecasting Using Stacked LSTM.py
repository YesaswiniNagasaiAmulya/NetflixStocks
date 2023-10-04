
# coding: utf-8

# In[70]:


### Data Collection
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# In[71]:


df=pd.read_csv("Netflix.csv")
print(df.head())


# In[72]:


closed_prices=df["Close"]
seq_len=15
mm=MinMaxScaler()
scaled_price=mm.fit_transform(np.array(closed_prices)[...,None]).squeeze()
print(scaled_price)


# In[79]:


x=[]
y=[]
for i in range(len(scaled_price)-seq_len):
    x.append(scaled_price[i:i+seq_len])
    y.append(scaled_price[i+seq_len])


# In[74]:


x=np.array(x)[...,None]
y=np.array(y)[...,None]

train_x = torch.from_numpy(x[:int(0.8 * x.shape[0])]).float()
train_y = torch.from_numpy(y[:int(0.8 * y.shape[0])]).float().unsqueeze(1)
test_x = torch.from_numpy(x[int(0.8 * x.shape[0]):]).float()
test_y = torch.from_numpy(y[int(0.8 * y.shape[0]):]).float().unsqueeze(1)
print(train_x.shape,test_x.shape)
print(train_y.shape,test_y.shape)


# In[76]:


class Model(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)
    def forward(self,x):
        output,(hidden,cell)=self.lstm(x)
        return self.fc(hidden[-1,:])
model=Model(1,150)

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()
num_epochs=110
for epoch in range(num_epochs):
    output=model(train_x)
    loss=loss_fn(output,train_y.squeeze(1))
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%10==0 and epoch!=0:
        print(epoch,"epoch loss",loss.item())
        
model.eval()
with torch.no_grad():
    output=model(test_x)
    
    
pred=mm.inverse_transform(output.numpy())
test_y_2d=test_y.reshape(-1,1)
real=mm.inverse_transform(test_y_2d.numpy())

plt.plot(pred.squeeze(),color="red",label="predicted")
plt.plot(real.squeeze(),color="green",label="real")
plt.show()

