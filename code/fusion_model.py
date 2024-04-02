import torch
from torch import nn
from lstm_model import LSTMModel
from densenet_model import densenet121

class FusionModel(nn.Module):
    def __init__(self, lstm_model, densenet_model):
        super(FusionModel, self).__init__()
        self.lstm_model = lstm_model
        self.densenet_model = densenet_model
        #self.linear = None  # 初始化为None，稍后会根据输出的维度来调整
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 添加设备属性
        self.linear = nn.Linear(in_features=40, out_features=1)
        self.linear.to(self.device)
        self.lstm_weight=0.7
        self.densenet_weight=0.3

    def forward(self, lstm_inputs, handcrafted_feature,dense_inputs):
        lstm_output = self.lstm_model(lstm_inputs, handcrafted_feature)
        densenet_output = self.densenet_model(dense_inputs)
       
        #fused_output = torch.cat((lstm_output, densenet_output), dim=1)
        #lstm_output = lstm_output.view(lstm_output.size(0), -1)  # 将LSTM输出展平为2D张量
        #densenet_output = densenet_output.view(densenet_output.size(0), -1)  # 将DenseNet输出展平为2D张量

        # 进行加权平均
        #weighted_output = self.lstm_weight * lstm_output + self.densenet_weight * densenet_output
        # 在维度1上连接两个张量
        combined_output = torch.cat((lstm_output, densenet_output), dim=1)
         # 在第一次前向传播时动态调整linear层的输入特征维度
        #if self.linear is None:
        #    combined_feature_size = weighted_output.shape[1]
        #    #print("combined_feature_size=",combined_feature_size)
        #    self.linear = nn.Linear(in_features=combined_feature_size, out_features=1)
        #    self.linear.to(self.device)
        #print("weighted_output=",weighted_output.shape)
        predictions = self.linear(combined_output)
        return predictions

