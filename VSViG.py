import torch
import torch.nn as nn
from torchvision import transforms
from timm.models.registry import register_model
import numpy as np
    
class InterPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
    def forward(self,x):
        B, C, P, _ = x.shape # BxT, C, P, 1
        tmp_x = x
        x_i = x.repeat(1,1,1,P) # BxT, C, P, P
        x_j = x_i
        for k in range(P):
            x_j[:,:,:,k] = x_i[:,:,k,k].unsqueeze(-1).repeat(1,1,P)
        relative = x_j - x_i
        for part in range(5):
            tmp_relative = relative
            tmp_relative[:,:,:,part*3:(part+1)*3] = relative[:,:,:,part*3:(part+1)*3] - 1e4
            tmp_x_j,_ = torch.max(tmp_relative, -1, keepdim=True)
            tmp_x[:,:,part*3:(part+1)*3,:] = tmp_x_j[:,:,part*3:(part+1)*3,:]
            
        x = torch.cat([x, tmp_x],1)
        return self.nn(x)

class IntraPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
    def forward(self,x):
        B, C, P, _ = x.shape # BxT, C, P, 1
        tmp_x = x
        x_i = x.repeat(1,1,1,P) # BxT, C, P, P
        x_j = x_i
        for k in range(P):
            x_j[:,:,:,k] = x_i[:,:,k,k].unsqueeze(-1).repeat(1,1,P)
        
        relative = x_j - x_i # BxT, C, P, P
        part = 1
        for point in range(P):
            tmp_x_j,_= torch.max(relative[:,:,point,(part-1)*3+1:part*3+1], -1, keepdim=True)
            # Part_x_j[:,:,point,1] = tmp_x_j.squeeze(-1)
            tmp_x[:,:,point,:] = tmp_x_j
            if point+1 % 3 == 0:
                part = 1+part
        x = torch.cat([x, tmp_x],1)
        return self.nn(x)

class Stem(nn.Module):

    def __init__(self, input_dim=3, output_dim=None, patch_size=32): # 32
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size), # 8
            nn.BatchNorm2d(output_dim),
        )
    def forward(self, x):
        B, T, P, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.stem(x) # BxTxP, C, 1, 1
        x = x.view(B, T, P, x.shape[1]) # B, T, P, C
        return x
    
class Stem_pe(nn.Module):

    def __init__(self, input_dim=3, output_dim=None, patch_size=32): # 32
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1), # 8
            nn.BatchNorm2d(output_dim)
        )
    def forward(self, x):
        B, T, P, C = x.shape
        x = x.view(-1, C, 1, 1)
        x = self.stem(x) # BxTxP, C, 1, 1
        x = x.view(B, T, P, x.shape[1]) # B, T, P, C
        return x
    
class Grapher(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn_inter = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels*5, 1, groups=4),
            nn.BatchNorm2d(out_channels*5),
            nn.ReLU()
            )
        self.nn_intra = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.fc4 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.InterPartMR = InterPartMR(out_channels)
        self.IntraPartMR = IntraPartMR(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # B, T, P, C = x.shape
        B, T, C, P, _ = x.shape # B,T,C,P,1
        x = x.view(-1,C,P,1) # BxT, C, P, 1
        tmp_x = x
        x = self.fc1(x)
        x = self.InterPartMR(x) # BxT, C*5, P, 1
        x = self.fc2(x)
        x = x+tmp_x
        x = self.act(x)
        x = self.fc3(x)
        x = self.IntraPartMR(x)

        x = self.fc4(x)
        x = x + tmp_x
        x = self.act(x)
        return x.view(B,T,C,P,1)

class Part_3DCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, dynamic=False, dynamic_point_order=None, SEED=None, expansion=4):
        super().__init__()
        self.expansion = expansion 
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), stride=stride, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels*self.expansion, 1),
            nn.BatchNorm3d(out_channels*self.expansion),
            nn.ReLU()
        )

        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels*self.expansion, 1, stride=stride),
                nn.BatchNorm3d(out_channels*self.expansion)
            )

        self.dynamic = dynamic
        # self.dynamic_part = self.dynamic_trans()
        self.dynamic_point_order = dynamic_point_order
        self.act = nn.ReLU()
        self.stride = stride
        self.in_ = in_channels
        self.SEED = SEED
    def dynamic_trans(self, x): # B,C,T,P,1, P: 15
        B,C,T,P,_ = x.shape
        x = x.view(-1,P)
        dynamic_order = self.dynamic_point_order[self.SEED]
        raw_order = list(np.arange(15))
        x[:,raw_order] = x[:,dynamic_order]

        return x.view(B,C,T,P,1)
    
    def forward(self, x):
        B,T,C,P,_ = x.shape # B,T,C,P,1
        x = x.transpose(1,2).contiguous() # B,C,T,P,1
        if self.dynamic:
            x = self.dynamic_trans(x)
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[:,:,:,:,1].unsqueeze(-1)
        x = self.conv3(x)
        residual = self.downsample(residual) 
        x = residual + x
        x = self.act(x)
        return  x.transpose(1,2).contiguous() # B, T, C*expansion, P, 1

class STViG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        height = 1080
        width = 1920
        T = 30
        points = 15
        # scale = 1/4
        dynamic = opt.dynamic
        num_layer = opt.num_layer
        output_channels = opt.output_channels
        dynamic_point_order = opt.dynamic_point_order
        expansion = opt.expansion
        self.pos_emb = opt.pos_emb
        if opt.pos_emb == 'add':
            ch4stem = output_channels[0] - 3
        else:
            ch4stem = output_channels[0]
        self.stem = Stem(input_dim=3, output_dim=ch4stem) # B T P C
        self.stem_pe = Stem_pe(input_dim=3, output_dim=ch4stem)
        self.in_channels = output_channels[0]
        self.backbone = []
        for stage in range(len(num_layer)):
            if stage > 0:
                self.backbone.append(Grapher(in_channels=self.in_channels, out_channels=output_channels[stage]))
                self.backbone.append(Part_3DCNN(stride=(2,1,1),
                                                in_channels= self.in_channels,
                                                out_channels=output_channels[stage],
                                                dynamic=dynamic,
                                                dynamic_point_order=dynamic_point_order,
                                                expansion= expansion,
                                                SEED=stage*num_layer[stage]+layers))
                self.in_channels = output_channels[stage] * expansion

            for layers in range(num_layer[stage]):
                self.backbone.append(Grapher(in_channels=self.in_channels, out_channels=output_channels[stage]))
                self.backbone.append(Part_3DCNN(in_channels=self.in_channels,
                                                out_channels=output_channels[stage],
                                                dynamic=dynamic,
                                                dynamic_point_order=dynamic_point_order,
                                                expansion=expansion,
                                                SEED=stage*num_layer[stage]+layers))
                if stage == 0:
                    self.in_channels = output_channels[stage] * expansion
                    
            
        self.backbone = nn.Sequential(*self.backbone)
        
        self.fc = nn.Sequential(nn.Conv2d(output_channels[-1] * expansion, 256, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256, 1, 1) # 2 classes
                                ) 

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
    
    def pe(self, flag, x, kpts=None):
        B, T, P, C = x.shape
        if flag == 'learn':
            x = x + nn.Parameter(torch.zeros(1, T, P, C)).cuda()
        elif flag == 'add':
            x = torch.cat((x,kpts),axis=-1)
        elif flag == 'stem':
            pe = self.stem_pe(kpts)
            x = x+pe
        elif flag == 'no':
            x = x
        return x

    def forward(self, inputs, kpts=None):
        # inputs: Batches, Frames, Points, Channles, Height, Width
        x = self.stem(inputs)
        x = self.pe(self.pos_emb, x, kpts)
        B, T, P, C = x.shape # Batches, Time, Points, Channles
        x = x.transpose(2,3).contiguous().view(B,T,C,P,1)
        x = self.backbone(x)
        B,T,C,P,_ = x.shape
        x = x.transpose(1,2).contiguous().view(B,C,T,P)
        x = nn.functional.adaptive_avg_pool2d(x, 1) # B, C, 1, 1
        return torch.sigmoid(self.fc(x).squeeze(-1).squeeze(-1).squeeze(-1))

@register_model
def VSViG_base(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, **kwargs):
            self.dynamic = 1
            self.num_layer = [2,2,6,2]
            self.output_channels = [24,48,96,192]
            self.dynamic_point_order = torch.load('PATH_TO_DYNAMIC_PARTITIONS')
            self.expansion = 2
            self.pos_emb = 'stem'
    opt = OptInit(**kwargs)
    model = STViG(opt)
    return model

@register_model
def VSViG_light(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, **kwargs):
            self.dynamic = 1
            self.num_layer = [2,2,6,2]
            self.output_channels = [12,24,48,96]
            self.dynamic_point_order = torch.load('PATH_TO_DYNAMIC_PARTITIONS')
            self.expansion = 2
            self.pos_emb = 'stem'
    opt = OptInit(**kwargs)
    model = STViG(opt)
    return model