import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class DummyLayer(nn.Module):
    def forward(self, x):
        return x


class CameraBranch(nn.Module):
    def __init__(self, flags):
        super(CameraBranch, self).__init__()
        if flags.base_network == 'GoogleNet':
            self.backbone = models.googlenet(pretrained=True)
            self.backbone.dropout = DummyLayer()
            self.fc = nn.Linear(2048, 128)
        elif flags.base_network == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.fc = nn.Linear(1024, 128)
        elif flags.base_network == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.fc = nn.Linear(4096, 128)
        else:
            raise NotImplementedError
        self.backbone.fc = DummyLayer()
        if flags.loss_fn == 'R':
            self.trans_branch = nn.Linear(128, 3)
            self.rot_branch = nn.Linear(128, 4)
        elif flags.loss_fn == 'C':
            self.TRANS_CLASS_NUM = int(flags.kmeans_trans_path.split('.')[-2].split('_')[-1])
            self.ROTS_CLASS_NUM = int(flags.kmeans_rots_path.split('.')[-2].split('_')[-1])
            self.trans_branch = nn.Linear(128, self.TRANS_CLASS_NUM)
            self.rot_branch = nn.Linear(128, self.ROTS_CLASS_NUM)

    def forward(self, img1, img2):
        im_feature1 = self.backbone(img1).flatten(start_dim=1)
        im_feature2 = self.backbone(img2).flatten(start_dim=1)
        im_feature = torch.cat((im_feature1, im_feature2), dim=1)
        im_feature = self.fc(im_feature)
        im_feature = F.relu(im_feature)
        trans = self.trans_branch(im_feature)
        rot = self.rot_branch(im_feature)
        output = {}
        output['tran'] = trans
        output['rot'] = rot
        return output

    def regression_loss(self, pred, gt):
        loss = nn.MSELoss()
        self.tran_loss = loss(pred['tran'], gt['tran'])
        self.rot_loss = loss(pred['rot'], gt['rot'])
        self.total_loss = self.tran_loss + self.rot_loss
        return self.total_loss, self.tran_loss, self.rot_loss

    def classification_loss(self, pred, tran_cls, rot_cls):
        celoss = nn.CrossEntropyLoss()
        self.tran_loss = celoss(pred['tran'], tran_cls.squeeze(1))
        self.rot_loss = celoss(pred['rot'], rot_cls.squeeze(1))
        self.total_loss = self.tran_loss + self.rot_loss
        return self.total_loss, self.tran_loss, self.rot_loss

