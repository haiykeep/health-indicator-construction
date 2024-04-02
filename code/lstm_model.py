import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=10, hidden_size=50, num_layers=1)#更改输入特征维度从17改为10
        self.attenion = Attention3dBlock()
        self.linear = nn.Sequential(
            nn.Linear(in_features=1500, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(inplace=True)
        )
        self.handcrafted = nn.Sequential(
            nn.Linear(in_features=20, out_features=10),#更改传感器数量从17改为10，特征由34变换为20
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, inputs, handcrafted_feature):
        y = self.handcrafted(handcrafted_feature)
        x, (hn, cn) = self.lstm(inputs)
        x = self.attenion(x)
        # flatten
        x = x.reshape(-1, 1500)
        x = self.linear(x)
        out = torch.cat((x, y), dim=1)
        #out = self.output(out)
        return out


class Attention3dBlock(nn.Module):
    def __init__(self):
        super(Attention3dBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=30, out_features=30),
            nn.Softmax(dim=2),
        )

    # inputs: batch size * window size(time step) * lstm output dims
    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.linear(x)
        x_probs = x.permute(0, 2, 1)
        # print(torch.sum(x_probs.item()))
        output = x_probs * inputs
        return output
