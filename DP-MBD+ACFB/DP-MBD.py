
import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops 
     
import torch     
import torch.nn as nn 

class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):   
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)  
        self.batch_norm = nn.BatchNorm2d(out_channels)    

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2] 
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)    
        return x


class DP-MBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)

        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        c = x
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        m = x
        c = self.cut_c(c)
        x = self.conv_x(x)
        x = self.act_x(x)
        x = self.batch_norm_x(x)
        m = self.max_m(m)
        m = self.batch_norm_m(m)
        x = torch.cat([c, x, m], dim=1)
        x = self.fusion(x)
        return x

 

