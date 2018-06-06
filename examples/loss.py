### Weighted F Score
def lyft_score(pred, target, weight):
    num,c,h,w = pred.shape
    pred = pred.view(num, c, -1)  # Flatten
    target = target.view(num, c, -1)  # Flatten
    intersection = (pred * target)
    int_sum = intersection.sum(dim=-1)
    pred_sum = pred.sum(dim=-1)
    targ_sum = target.sum(dim=-1)
    
    eps = 1e-9
    precision = int_sum / (pred_sum + eps)
    recall = int_sum / (targ_sum + eps)
    beta = V(weight ** 2)
    
    fnum = (1.+beta) * precision * recall
    fden = beta * precision + recall + eps
    
    fscore = fnum / fden
    
#     fb = (precision*recall)/precision*beta + recall + eps
    
    avg_w = torch.cuda.FloatTensor([0,.5,.5])
    favg = V(avg_w) * fscore
#     pdb.set_trace()
    return favg.sum(dim=-1)

class FLoss(nn.Module):
    def __init__(self, weight=torch.cuda.FloatTensor([1,2,0.5]), softmax=True):
        super().__init__()
        self.weight = weight
        self.softmax = softmax

    def forward(self, logits, targets):
        probs = F.softmax(logits) if self.softmax else F.sigmoid(logits)
        num = targets.size(0)  # Number of batches
        targets = torch.cat(((targets==0).unsqueeze(1), (targets==1).unsqueeze(1), (targets==2).unsqueeze(1)), dim=1).float()
        if isinstance(logits.data, torch.cuda.HalfTensor):
            targets = targets.half()
        else:
            targets = targets.float()
            
        score = lyft_score(probs, targets, self.weight)
        score = 1 - score.sum() / num
        return score


### Weighted Dice loss
def dice_coeff_weight(pred, target, weight):
    smooth = 1.
    num,c,h,w = pred.shape
    m1 = pred.view(num, c, -1)  # Flatten
    m2 = target.view(num, c, -1)  # Flatten
    intersection = (m1 * m2)
    w = V(weight.view(1,-1,1))
    i_w = (w*intersection).sum()
    m1_w = (w*m1).sum()
    m2_w = (w*m2).sum()
    return (2. * i_w + smooth) / (m1_w + m2_w + smooth)

def dice_coeff(pred, target):
    smooth = 1.
    num,c,h,w = pred.shape
    m1 = pred.view(num, c, -1)  # Flatten
    m2 = target.view(num, c, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, softmax=True):
        super(SoftDiceLoss, self).__init__()
        self.weight = weight
        self.softmax = softmax

    def forward(self, logits, targets):
        probs = F.softmax(logits) if self.softmax else F.sigmoid(logits)
        num = targets.size(0)  # Number of batches
        targets = torch.cat(((targets==0).unsqueeze(1), (targets==1).unsqueeze(1), (targets==2).unsqueeze(1)), dim=1).float()
        if isinstance(logits.data, torch.cuda.HalfTensor):
            targets = targets.half()
        else:
            targets = targets.float()
        if self.weight is not None:
            score = dice_coeff_weight(probs, targets, self.weight)
        else:
            score = dice_coeff(probs, targets)
        score = 1 - score.sum() / num
        return score


#### Weighted Cross-Entropy Loss
torch.nn.CrossEntropyLoss(weight=class_weights)