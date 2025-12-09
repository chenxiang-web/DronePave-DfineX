import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops  
    
import torch  
import torch.nn as nn
from torch.nn import init 
     
from engine.extre_module.ultralytics_nn.conv import Conv

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):     
        super().__init__()     
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),    
            nn.Sigmoid()   
        )

    def init_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear): 
                init.normal_(m.weight, std=0.001)    
                if m.bias is not None:   
                    init.constant_(m.bias, 0)    

    def forward(self, x):
        b, c, _, _ = x.size()    
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ACFB(nn.Module):  
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.adjust_conv = nn.Identity()   
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        
        self.se = SEAttention(inc[1] * 2)  
     
        if (inc[1] * 2) != ouc:   
            self.conv1x1 = Conv(inc[1] * 2, ouc) 
        else:  
            self.conv1x1 = nn.Identity()
    
    def forward(self, x):
        def forward(self, x):
        x1, x2 = x
        x1 = self.adjust_conv(x1)
        x_concat = torch.cat([x1, x2], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x1_weight, x2_weight = torch.split(x_concat, [x1.size()[1], x2.size()[1]], dim=1)
        x1_weight = x1 * x1_weight
        x2_weight = x2 * x2_weight
        return self.conv1x1(torch.cat([x1 + x2_weight, x2 + x1_weight], dim=1))
        


