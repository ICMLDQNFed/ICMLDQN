import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_KL_LS_loss(nn.Module):
    def __init__(self, n_classes, params = [1, .9, .07], eps = .3):
        super(CE_KL_LS_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.C = n_classes
        self.params = params
        self.eps = eps

    def smooth(self, target):
        out = (self.eps / (self.C - 1)) * torch.ones((target.shape[0], self.C))
        for row, col in enumerate(target.cpu()):
            out[row, col] = 1 - self.eps
        out = out.to(target.device)
        return out

    def forward(self, output, target, centroids):
        rttensor=torch.zeros(3).to(target.device)
        loss = self.ce_loss(output, target)
        surrogate_loss1 = F.kl_div(centroids.log(), F.log_softmax(output,1), reduction="batchmean", log_target=True)
        smooth_label = self.smooth(target)
        surrogate_loss2 = F.kl_div(smooth_label.log(), F.log_softmax(output, 1), reduction = 'batchmean', log_target=True)
        rttensor[0], rttensor[1], rttensor[2] = loss, surrogate_loss1, surrogate_loss2
        return self.params[0] * loss + self.params[1] * surrogate_loss1 + self.params[2] * surrogate_loss2