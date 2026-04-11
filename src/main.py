import torch 
import torch.nn as nn
import pandas as pd
from pathlib import Path

# Carregar os dados
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "telco_churn.csv"

df= pd.read_csv("data/raw/telco_churn.csv")


Y=df['Churn'].map({"Yes": 1, "No": 0})
X=df.drop(columns=['Churn', 'customerID'])

X = pd.get_dummies(X, drop_first=True)

X = X.values    
y = Y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train = torch.tensor(X_train,dtype=torch.float32)
X_test  = torch.tensor(X_test,dtype=torch.float32)

y_train = torch.tensor(X_train,dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)



# Pré-processamento dos dados
class RegressaoLogistic(nn.Module):
    def __init__(self, input_dim):
        super(RegressaoLogistic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)
    

model= RegressaoLogistic(input_dim=10)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
    logits= model(X_train)
    loss=  loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

