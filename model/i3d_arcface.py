import torch
import torch.nn as nn

from model.backbone_2d import iresnet50

def conv3x3x3(in_planes, out_planes, temporal_stride=1, spatial_stride=1):
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(3, 3, 3),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(1, 1, 1),
                     bias=False)

def conv1x1x1(in_planes, out_planes, temporal_stride=1, spatial_stride=2):
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(1, 1, 1),
                     stride=(temporal_stride, spatial_stride, spatial_stride))

class IBasicBlock(nn.Module):
    block_expansion = 1
    def __init__(self, inplanes, planes, temporal_stride=1, spatial_stride=1, downsample=None):
        super(IBasicBlock, self).__init__()
        self.eps = 1e-05
        self.bn1 = nn.BatchNorm3d(inplanes, eps=self.eps)
        self.conv1 = conv3x3x3(inplanes, planes, 1, 1)
        self.bn2 = nn.BatchNorm3d(planes, eps=self.eps)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3x3(planes, planes, temporal_stride, spatial_stride)
        self.bn3 = nn.BatchNorm3d(planes, eps=self.eps)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    
class I3D_ResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, layers, backbone_2d_path, dropout=0, num_features=512):
        super(I3D_ResNet, self).__init__()
        self.inplanes = 64
        self.eps = 1e-05
        self.conv1 = conv3x3x3(3, self.inplanes)
        self.bn1 = nn.BatchNorm3d(self.inplanes, eps=self.eps)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], spatial_stride=2)
        self.layer2 = self._make_layer(128, layers[0], spatial_stride=2)
        self.layer3 = self._make_layer(256, layers[0], spatial_stride=2)
        self.layer4 = self._make_layer(512, layers[0], spatial_stride=2)
        self.bn2 = nn.BatchNorm3d(512, eps=self.eps)
        self.gpool3d = nn.AdaptiveMaxPool3d(output_size=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=self.eps)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.inflate_weights(backbone_2d_path)
        
    def _make_layer(self, planes, num_block,
                    temporal_stride=1, spatial_stride=1):
        downsample = None
        if spatial_stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes, temporal_stride=temporal_stride, spatial_stride=spatial_stride),
                nn.BatchNorm3d(planes, eps=self.eps)
            )
        layers = []
        layers.append(
            IBasicBlock(self.inplanes, planes, temporal_stride=1, spatial_stride=2, downsample=downsample)
        )
        self.inplanes = planes
        for _ in range(1, num_block):
            layers.append(
                IBasicBlock(self.inplanes, planes)
            )
        
        return nn.Sequential(*layers)
    
    def inflate_weights(self, backbone_2d_path):
        R2D = iresnet50(pretrained=backbone_2d_path)
        def _copy_params(src: nn.Module, dst: nn.Module, weight=True, bias=False, running_mean=False, running_var=False, weight_copy=1, is_conv=True):
            if weight:
                if is_conv:
                    dst.weight.data.copy_(torch.unsqueeze(src.weight.data, dim=2).repeat(1, 1, weight_copy, 1, 1))
                    dst.weight.data = dst.weight.data / weight_copy
                else:
                    dst.weight.data.copy_(src.weight.data)
            if bias:
                dst.bias.data.copy_(src.bias.data)
            if running_mean:
                dst.running_mean.data.copy_(src.running_mean.data)
            if running_var:
                dst.running_var.data.copy_(src.running_var.data)
        
        _copy_params(R2D.conv1, self.conv1, weight_copy=3)
        _copy_params(R2D.bn1, self.bn1, bias=True, running_mean=True, running_var=True, is_conv=False)
        _copy_params(R2D.prelu, self.prelu, is_conv=False)

        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        R2D_layers = [R2D.layer1, R2D.layer2, R2D.layer3, R2D.layer4]

        for s, _ in enumerate(stages):
            res = stages[s]._modules
            count = 0
            for id, dst in res.items():
                _copy_params(R2D_layers[s]._modules[str(id)].conv1, dst.conv1, weight_copy=3)
                _copy_params(R2D_layers[s]._modules[str(id)].conv2, dst.conv2, weight_copy=3)
                _copy_params(R2D_layers[s]._modules[str(id)].bn1, dst.bn1, bias=True, running_mean=True, running_var=True, is_conv=False)
                _copy_params(R2D_layers[s]._modules[str(id)].bn2, dst.bn2, bias=True, running_mean=True, running_var=True, is_conv=False)
                _copy_params(R2D_layers[s]._modules[str(id)].bn3, dst.bn3, bias=True, running_mean=True, running_var=True, is_conv=False)
                _copy_params(R2D_layers[s]._modules[str(id)].prelu, dst.prelu, is_conv=False)
                if dst.downsample is not None:
                    down_conv = dst.downsample._modules['0']
                    down_bn = dst.downsample._modules['1']
                    _copy_params(R2D_layers[s]._modules[str(id)].downsample._modules['0'], down_conv, weight_copy=1)
                    _copy_params(R2D_layers[s]._modules[str(id)].downsample._modules['1'], down_bn, bias=True, running_mean=True, running_var=True, is_conv=False)
                count += 1
        _copy_params(R2D.bn2, self.bn2, bias=True, running_mean=True, running_var=True, is_conv=False)
        print('I3D params inflated from pretrained ResNet')
    
    def forward(self, x):
        bs, _, _, _, _ = x.shape
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn2(out)
        pooled = self.gpool3d(out)
        out = pooled.view(bs, -1)
        # out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.features(out)
        return out
