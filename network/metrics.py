import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

#TODO:デフォルトパラメータを論文推奨どおりに変更
class SoftMaxLayer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftMaxLayer, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.fc_l = nn.Linear(num_features, num_classes)
        self.fc_w = nn.Parameter(self.fc_l.weight)

    def forward(self, input, label=None):
        logits = self.fc_l(input)
        theta = torch.ones(self.num_classes) #他と出力個数を揃えるためのダミー
        s = torch.ones(1) #他と出力個数を揃えるためのダミー
        if label is None:
            return logits, theta, input, self.fc_w

        return logits, theta, s, input, self.fc_w


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.0):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        if label is None:
            return logits, theta, x, W

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            #print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output, theta, self.s, x, W


class ArcFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        if label is None:
            return logits, theta, x, W

        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output, theta, self.s * torch.ones(1), x, W


class SphereFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        if label is None:
            return logits, theta, x, W

        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output, theta, self.s*torch.ones(1), x, W


class CosFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))

        if label is None:
            return logits, theta, x, W 

        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output, theta, self.s * torch.ones(1), x, W 

class Okatani(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Okatani, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(num_features, 1)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        # feature re-scale
        scale = torch.exp(self.bn_scale(self.fc_scale(input)))
        logits *= scale #岡谷研方式のみsをかけた状態で出力(他の方式ではsをかける前)
        if label is None:
            return logits, theta, x, W 

        return logits, theta, torch.ones(1), x, W #sはダミーで1とする
