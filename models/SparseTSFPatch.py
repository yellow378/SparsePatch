import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * self.period_len // 2,
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        #my add begin
        self.patchEmbeding = nn.Sequential(
            nn.Linear(self.period_len, self.d_model),
            nn.ReLU()
        )

        
        self.linear = nn.Linear(self.seg_num_x, self.d_model, bias=False)

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.period_len)
        )
        #my add end

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x


        # downsampling: b,c,s -> bc,n,w 
        x = x.reshape(-1, self.seg_num_x, self.period_len)

        #Batch bc,n,d_model -> bc,d_model,n
        x = self.patchEmbeding(x).permute(0,2,1)

        # sparse forecasting bc,d_model,m -> bc, m ,d_model
        y = self.linear(x).permute(0,2,1)

        #prediction bc,m,w
        y = self.predict(y)
        # upsampling: bc,m,w -> b,c,s
        y = y.reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
