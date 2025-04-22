import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.

        Parameters
        ----------
        gamma : float, default=2
            Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma.
        alpha : float or tensor or list, optional
            Balancing factor. For binary/multi-label, a scalar in [0,1]. For multi-class, a
            tensor or list of per-class weights. If None, no class balancing is applied.
        reduction : {'none', 'mean', 'sum'}, default='mean'
            Specifies the reduction to apply to the output: no reduction, mean or sum.
        task_type : {'binary', 'multi-class', 'multi-label'}, default='binary'
            Type of classification task for which to compute the loss.
        num_classes : int, optional
            Number of classes; required if task_type=='multi-class' and alpha is provided.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Dispatches to the appropriate method based on self.task_type.

        Parameters
        ----------
        inputs : Tensor
            Model outputs (logits). Shape:
            - binary/multi-label: (batch_size, num_classes)
            - multi-class:         (batch_size, num_classes)
        targets : Tensor
            Ground truth labels. Shape:
            - binary:      (batch_size,)
            - multi-label:(batch_size, num_classes)
            - multi-class:(batch_size,) or one-hot of shape (batch_size, num_classes)

        Returns
        -------
        Tensor
            Loss value per sample or reduced according to `self.reduction`.
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """
        Focal loss for binary classification.

        Parameters
        ----------
        inputs : Tensor
            Logits of shape (batch_size,).
        targets : Tensor
            Binary labels of shape (batch_size,), values in {0,1}.

        Returns
        -------
        Tensor
            Loss values reduced according to `self.reduction`.
        """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """
        Focal loss for multi-class classification.

        Parameters
        ----------
        inputs : Tensor
            Logits of shape (batch_size, num_classes).
        targets : Tensor
            One-hot encoded targets of shape (batch_size, num_classes).

        Returns
        -------
        Tensor
            Loss values reduced according to `self.reduction`.
        """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # # One-hot encode the targets
        targets_one_hot = targets
        targets = torch.argmax(targets_one_hot, dim=1)
        # targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """
        Focal loss for multi-label classification.

        Parameters
        ----------
        inputs : Tensor
            Logits of shape (batch_size, num_classes).
        targets : Tensor
            Binary indicator tensor of shape (batch_size, num_classes).

        Returns
        -------
        Tensor
            Loss values reduced according to `self.reduction`.
        """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
