import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class SFU(nn.Module):
    def __init__(self,out_channels):
        super(SFU,self).__init__()
        self.out_channels = 2 * out_channels
        
    
    def forward(self,x_RGB,x_DOL):
        out = torch.cat((x_RGB,x_DOL),1)
        return out
        

class GFU(nn.Module):
    def __init__(self,out_channels):
        super(GFU,self).__init__()
        self.input_channels = out_channels
        self.out_channels = out_channels

        self.conv_RGB = nn.Conv2d(self.input_channels,2*self.input_channels,kernel_size=3,padding=1)
        self.act_RGB = nn.ReLU()
        
        self.conv_DOL = nn.Conv2d(self.input_channels,2*self.input_channels,kernel_size=3,padding=1)
        self.act_DOL = nn.ReLU()
        
        self.conv_MIX = nn.Conv2d(4*self.input_channels,self.out_channels,kernel_size=1)
        self.act_MIX = nn.ReLU()
        

    def forward(self,x_RGB,x_DOL):
        x_concat_1 = torch.cat((x_RGB,x_DOL),1)

        x_RGB = self.conv_RGB(x_RGB)
        x_RGB = self.act_RGB(x_RGB)
        
        x_DOL = self.conv_DOL(x_DOL)
        x_DOL = self.act_DOL(x_DOL)
        
        x_RGB = x_RGB.add(x_concat_1)
        x_DOL = x_DOL.add(x_concat_1)
        
        x_concat_2 = torch.cat((x_RGB,x_DOL),1)

        x_MIX = self.conv_MIX(x_concat_2)
        x_MIX = self.act_MIX(x_MIX)
        
        
        return x_MIX
        
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256,dataset=None):
        super(ClassificationModel, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        if self.dataset is None:
            self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
            self.output_act = nn.Sigmoid()
        else:
            setattr(self,'output_{}'.format(self.dataset),nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1))
            setattr(self,'output_act_{}'.format(self.dataset),nn.Sigmoid())


    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)
        if self.dataset is None:
            out = self.output(out)
            out = self.output_act(out)
        else:
            out = getattr(self,'output_{}'.format(self.dataset))(out)
            out = getattr(self,'output_act_{}'.format(self.dataset))(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape
       
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, evaluate=False,ignore_class=False,dataset=None):
        if ignore_class:
            num_classes -= 1
            self.ignore_index = num_classes
        else:
            self.ignore_index = None
        self.dataset = dataset
        self.evaluate=evaluate
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes,dataset=dataset)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        if self.dataset is None:
          self.classificationModel.output.weight.data.fill_(0)
          self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        else:
          with torch.no_grad():
            getattr(self.classificationModel,'output_{}'.format(self.dataset)).weight.data.fill_(0)
            getattr(self.classificationModel,'output_{}'.format(self.dataset)).bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, False, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, False))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if not self.evaluate:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        #TMP
        classification = torch.Tensor(np.zeros((1,181089,8),dtype=np.float32)).cuda()
        regression = torch.Tensor(np.zeros((1,181089,4),dtype=np.float32)).cuda()
        for i in [10000]: #[ 8524,  8528,  9982,  9990,  9991,  9992,  9993,  9994,  9999, 10000, 10001, 10002, 10003, 10004, 10008, 10009, 10010, 10011, 10012, 10018, 11476, 11480]:
          classification[0,i,0]=1

        anchors = self.anchors(img_batch)
        
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > 0.05)[0, :, 0]
               
        classification_2 = classification[:, scores_over_thresh, :]
        
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

        nms_scores, nms_class = classification_2[0, anchors_nms_idx, :].max(dim=1)
        
        return classification, regression, anchors, annotations, (nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :])

def resnet18(num_classes, pretrained=False, color_mode='RGB', fusion_type=0, step=1, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if step==2:
        model = Double_ResNet(num_classes, BasicBlock, [2, 2, 2, 2], color_mode, fusion_type, step, **kwargs)
    else:
        model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, color_mode='RGB', fusion_type=0, step=1, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if step==2:
        model = Double_ResNet(num_classes, BasicBlock, [3, 4, 6, 3], color_mode, fusion_type, step, **kwargs)
    else:
        model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, color_mode='RGB', fusion_type=0, step=1, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if step==2:
        model = Double_ResNet(num_classes, Bottleneck, [3, 4, 6, 3], color_mode, fusion_type, step, **kwargs)
    else:
        model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, color_mode='RGB', fusion_type=0, step=1, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if step==2:
        model = Double_ResNet(num_classes, Bottleneck, [3, 4, 23, 3], color_mode, fusion_type, step, **kwargs)
    else:
        model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, color_mode='RGB', fusion_type=0, step=1, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if step==2:
        model = Double_ResNet(num_classes, Bottleneck, [3, 8, 36, 3], color_mode, fusion_type, step, **kwargs)
    else:
        model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


#################################################################################################################

class Double_ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, color_mode, fusion_type, step, evaluate = False, ignore_class=False,dataset=None):
        
        if ignore_class:
            num_classes -= 1
            self.ignore_index = num_classes
        else:
            self.ignore_index = None
        self.dataset = dataset
        self.evaluate=evaluate
        self.fusion_type = fusion_type
        self.inplanes = [64,64]
        self.step=step
        if color_mode == 'ALL':
            self.color_mode = ['RGB','DOL']
        else:
            self.color_mode = [color_mode]
        super(Double_ResNet, self).__init__()
        
        # Backbone RGB
        if 'RGB' in self.color_mode:
            self.conv1_RGB = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1_RGB = nn.BatchNorm2d(64)
            self.relu_RGB = nn.ReLU(inplace=True)
            self.maxpool_RGB = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            if self.step==2:
                for parameters in [self.conv1_RGB.weight,self.bn1_RGB.weight, self.bn1_RGB.bias]:
                    parameters.requires_grad=False

            self.layer1_RGB = self._make_layer(block, 64, layers[0], 'RGB', self.step) 
            self.layer2_RGB = self._make_layer(block, 128, layers[1], 'RGB', self.step, stride=2)
            self.layer3_RGB = self._make_layer(block, 256, layers[2], 'RGB', self.step, stride=2)
            self.layer4_RGB = self._make_layer(block, 512, layers[3], 'RGB', self.step, stride=2)
        
        # Backbone DOL
        if 'DOL' in self.color_mode:
            self.conv1_DOL = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1_DOL = nn.BatchNorm2d(64)
            self.relu_DOL = nn.ReLU(inplace=True)
            self.maxpool_DOL = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            if self.step==2:
                for parameters in [self.conv1_DOL.weight,self.bn1_DOL.weight, self.bn1_DOL.bias]:
                    parameters.requires_grad=False

            self.layer1_DOL = self._make_layer(block, 64, layers[0], 'DOL', self.step)
            self.layer2_DOL = self._make_layer(block, 128, layers[1], 'DOL', self.step, stride=2)
            self.layer3_DOL = self._make_layer(block, 256, layers[2], 'DOL', self.step, stride=2)
            self.layer4_DOL = self._make_layer(block, 512, layers[3], 'DOL', self.step, stride=2)

        # FPN Sizes
        if 'RGB' in self.color_mode:
            if block == BasicBlock:
                conv_sizes = [self.layer2_RGB[layers[1] - 1].conv2.out_channels, self.layer3_RGB[layers[2] - 1].conv2.out_channels,
                             self.layer4_RGB[layers[3] - 1].conv2.out_channels]
            elif block == Bottleneck:
                conv_sizes = [self.layer2_RGB[layers[1] - 1].conv3.out_channels, self.layer3_RGB[layers[2] - 1].conv3.out_channels,
                             self.layer4_RGB[layers[3] - 1].conv3.out_channels]
            else:
                raise ValueError(f"Block type {block} not understood")
        elif 'DOL' in self.color_mode:
            if block == BasicBlock:
                conv_sizes = [self.layer2_DOL[layers[1] - 1].conv2.out_channels, self.layer3_DOL[layers[2] - 1].conv2.out_channels,
                             self.layer4_DOL[layers[3] - 1].conv2.out_channels]
            elif block == Bottleneck:
                conv_sizes = [self.layer2_DOL[layers[1] - 1].conv3.out_channels, self.layer3_DOL[layers[2] - 1].conv3.out_channels,
                             self.layer4_DOL[layers[3] - 1].conv3.out_channels]
            else:
                raise ValueError(f"Block type {block} not understood")
            
        # Fusion_types (1 : Stack Fusion Unit, 2 : Gated Fusion Unit)
        if self.fusion_type==1:
            self.FU_2 = SFU(conv_sizes[0])
            self.FU_3 = SFU(conv_sizes[1])
            self.FU_4 = SFU(conv_sizes[2])
        elif self.fusion_type==2:
            self.FU_2 = GFU(conv_sizes[0])
            self.FU_3 = GFU(conv_sizes[1])
            self.FU_4 = GFU(conv_sizes[2])
        elif self.fusion_type!=0:
            raise ValueError(f"Fusion type {fusion_type} not understood")
        
        
        if fusion_type!=0:
            self.fpn = PyramidFeatures(self.FU_2.out_channels, self.FU_3.out_channels, self.FU_4.out_channels)
        elif 'RGB' in self.color_mode:
            self.fpn = PyramidFeatures(conv_sizes[0], conv_sizes[1], conv_sizes[2])
        else:
            self.fpn = PyramidFeatures(conv_sizes[0], conv_sizes[1], conv_sizes[2])
            
            
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes,dataset=dataset)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        if self.dataset is None:
          self.classificationModel.output.weight.data.fill_(0)
          self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        else:
          with torch.no_grad():
            getattr(self.classificationModel,'output_{}'.format(self.dataset)).weight.data.fill_(0)
            getattr(self.classificationModel,'output_{}'.format(self.dataset)).bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()
        
    def _make_layer(self, block, planes, blocks, color, step, stride=1):
        if step ==2:
            freeze=True
        else:
            freeze=False
        downsample = None
        if color == 'RGB':
            ind=0
        elif color == 'DOL':
            ind=1
        if stride != 1 or self.inplanes[ind] != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes[ind], planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            if step==2:
                for parameters in [downsample[0].weight,downsample[1].weight,downsample[1].bias]:
                    parameters.requires_grad=False

        layers = [block(self.inplanes[ind], planes, freeze, stride, downsample)]
        self.inplanes[ind] = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes[ind], planes, freeze))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if not self.evaluate:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        # Backbone RGB
        if 'RGB' in self.color_mode:
            if 'DOL' in self.color_mode:
                if not self.evaluate:
                    x_RGB = self.conv1_RGB(img_batch[0]) 
                else:
                    x_RGB = self.conv1_RGB(img_batch[:,0:3,:,:])
            else:
                x_RGB = self.conv1_RGB(img_batch)
            x_RGB = self.bn1_RGB(x_RGB)
            x_RGB = self.relu_RGB(x_RGB)
            x_RGB = self.maxpool_RGB(x_RGB)

            x1_RGB = self.layer1_RGB(x_RGB)
            x2_RGB = self.layer2_RGB(x1_RGB)
            x3_RGB = self.layer3_RGB(x2_RGB)
            x4_RGB = self.layer4_RGB(x3_RGB)
        
        # Backbone DOL
        if 'DOL' in self.color_mode:
            if 'RGB' in self.color_mode:
                if not self.evaluate:
                    x_DOL = self.conv1_DOL(img_batch[1])
                else:
                    x_DOL = self.conv1_DOL(img_batch[:,3:6,:,:])
            else:
                x_DOL = self.conv1_DOL(img_batch)
            x_DOL = self.bn1_DOL(x_DOL)
            x_DOL = self.relu_DOL(x_DOL)
            x_DOL = self.maxpool_DOL(x_DOL)

            x1_DOL = self.layer1_DOL(x_DOL)
            x2_DOL = self.layer2_DOL(x1_DOL)
            x3_DOL = self.layer3_DOL(x2_DOL)
            x4_DOL = self.layer4_DOL(x3_DOL)

        if 'RGB' in self.color_mode and 'DOL' in self.color_mode:
            x2 = self.FU_2(x2_RGB,x2_DOL)
            x3 = self.FU_3(x3_RGB,x3_DOL)
            x4 = self.FU_4(x4_RGB,x4_DOL)
            features = self.fpn([x2, x3, x4])
            
        elif 'RGB' in self.color_mode:
            features = self.fpn([x2_RGB,x3_RGB,x4_RGB])

        elif 'DOL' in self.color_mode:
            features = self.fpn([x2_DOL,x3_DOL,x4_DOL])            
        
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        if 'RGB' in self.color_mode and 'DOL' in self.color_mode:
            if not self.evaluate:
                anchors = self.anchors(img_batch[0])
            else:
                anchors = self.anchors(img_batch[:,0:3,:,:])
        else:
            anchors = self.anchors(img_batch)
        if not self.evaluate:
            return self.focalLoss(classification, regression, anchors, annotations, ignore_index = self.ignore_index)
        
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            #print(nms_scores)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
