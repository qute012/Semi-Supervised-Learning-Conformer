import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, reduce='none', logit_temp=0.1):
        self.reduce = reduce
        self.logit_temp = logit_temp
        self.sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.criterion = nn.BCEWithLogitsLoss(reduce=reduce)

    def forward(self, x, positives, negatives):
        positives = positives.unsqueeze(0)
        samples = torch.cat([positives, negatives], dim=0)

        logits = self.sim(x.float(), samples.float()).type_as(x) / self.logit_temp
        logits = logits.transpose(0, 2).view(-1, logits.size(-1))

        target = logits.new_zeros(logits.size(1)*logits.size(2), dtype=torch.float)

        denom = x.size(0)
        loss = self.criterion(logits, target)

        if self.reduce=='none':
             return loss.sum() / denom
        else:
            return loss