from torch import nn
import torch


class RegLossForClassification(nn.Module):
    def __init__(self, alpha=0.5):
        super(RegLossForClassification, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss()
        
        self.alpha = alpha
    
    def forward(self, logits, labels):
        bs = logits.shape[0]
        ce_loss = self.cross_entropy(logits, labels)
        # batch_size, 6
        probs = self.softmax(logits)
        arange = torch.stack([torch.arange(logits.shape[1]) for _ in range(bs)]).to(probs.dtype).to(probs.device)
        reg_scores = (probs * arange).sum(dim=-1)
        labels_float = labels.to(probs.dtype).to(probs.device)
        mse_loss = self.mse(reg_scores, labels_float)
        
        loss = self.alpha * ce_loss / ce_loss.detach() + (1 - self.alpha) * mse_loss / mse_loss.detach()
        
        return loss