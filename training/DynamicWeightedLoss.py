import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightedLoss(nn.Module):
    """
    Dynamically balance presence and type losses based on their performance
    Similar to uncertainty weighting in multi-task learning
    """
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight_presence: float = 2.0,
        pos_weight_types: float = 1.5,
        adaptive: bool = True
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight_presence = pos_weight_presence
        self.pos_weight_types = pos_weight_types
        self.adaptive = adaptive
        
        # Learnable loss weights (log variance approach)
        if adaptive:
            self.log_var_presence = nn.Parameter(torch.zeros(1))
            self.log_var_type = nn.Parameter(torch.zeros(1))
    
    def focal_loss_with_logits(self, logits, targets, alpha, gamma, pos_weight):
        """Focal loss implementation"""
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor([pos_weight]).to(logits.device)
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** gamma
        
        if alpha >= 0:
            alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * bce_loss).mean()
    
    def forward(self, presence_logits, type_logits, presence_labels, type_labels):
        """
        Args:
            presence_logits: (batch, 1)
            type_logits: (batch, num_types)
            presence_labels: (batch,)
            type_labels: (batch, num_types)
        """
        # Presence loss
        presence_loss = self.focal_loss_with_logits(
            presence_logits.squeeze(-1),
            presence_labels.float(),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            pos_weight=self.pos_weight_presence
        )
        
        # Type loss (only for positive samples)
        has_aneurysm = (presence_labels > 0).float().unsqueeze(-1)
        
        if has_aneurysm.sum() > 0:
            type_loss = self.focal_loss_with_logits(
                type_logits,
                type_labels.float(),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                pos_weight=self.pos_weight_types
            )
            type_loss = type_loss * has_aneurysm.mean()
        else:
            type_loss = torch.tensor(0.0, device=presence_logits.device)
        
        # Dynamic weighting using uncertainty
        if self.adaptive:
            # Inverse variance weighting
            precision_presence = torch.exp(-self.log_var_presence)
            precision_type = torch.exp(-self.log_var_type)
            
            total_loss = (
                precision_presence * presence_loss + self.log_var_presence +
                precision_type * type_loss + self.log_var_type
            )
