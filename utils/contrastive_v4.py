from typing import Optional
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F
from utils.contrastive import ContrastManager
from scene.cameras import Camera

from scene.dataset_readers import SceneType


class ContrastManagerV4(ContrastManager):
    def __init__(
        self, 
        cfg: DictConfig, 
        example_cam: Camera, # A training camera 
        valid_mask_by_name: Optional[dict],
        scene_type: SceneType,
    ) -> None:
        super().__init__(cfg, example_cam, valid_mask_by_name, scene_type)

        print("Initializing ContrastiveManagerV4...")

    def contrastive_loss(
            self,
            features: torch.Tensor, 
            instance_labels: torch.Tensor, 
            temperature: float, 
            weight: Optional[torch.Tensor] = None,
            sum_in_log: bool = True, 
            sim_exp: float = 1.0,
            weighting_mode: str = "none",
        ):
        '''
        Args:
            features: (N, D), features of the sampled pixels
            instance_labels: (N, ) or (N, L), integer labels or multi-hot bool labels
            temperature: temperature for the softmax
            weight: (N, ), weight for each sample in the final loss
            sum_in_log: whether to sum in log space or not
            sim_exp: exponent for the similarity
            weighting_mode: "on_sim" or "on_prob"
        '''
        assert features.shape[0] == instance_labels.shape[0], f"{features.shape}, {instance_labels.shape} does not match. "
        if weight is not None:
            assert features.shape[0] == weight.shape[0], f"{features.shape}, {weight.shape} does not match. "
        
        bsize = features.size(0) # N
        if instance_labels.dim() != 1: # only supporting integer labels (N, )
            raise Exception(f"instance_labels.dim() {instance_labels.dim()} is not supported")
        
        # Compute a boolean mask of the positive pairs. Each entry (i, j) is True if instance_labels[i] 
        # and instance_labels[j] are the same class label.
        sim_masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone()) # (N, N)
        # Exclude self-comparison by zeroing out the diagonal
        sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)

        # compute similarity matrix based on Euclidean distance
        # distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1) # (N, N)

        # Pair‑wise squared Euclidean distance matrix
        # distance_sq[i,j] = ||f_i – f_j||_2
        distance_sq = (features.unsqueeze(1) - features.unsqueeze(0)).pow(2).sum(-1)  # (N, N)
        distance_sq = distance_sq.sqrt()  # (N, N) ‑ take the square root to get the distance

        # Mask out only the *positive* pairs (same instance, but i ≠ j)
        # pos_mask = sim_masks                               # (N, N) ‑ from code you already built
        # pos_distances = distance_sq[pos_mask]              # 1‑D tensor of all positive distances
        # pos_loss = pos_distances.mean()

        # # Push the negative pairs apart up to a margin
        # margin = 1.0
        # neg_mask = ~sim_masks
        # neg_distances = distance_sq[neg_mask]
        # neg_loss = (F.relu(margin - neg_distances)).pow(2).mean()

        # # Combine the losses constributed by positive pairs and negative pairs
        # loss = pos_loss + neg_loss


        # (N, N) boolean masks
        pos_mask = sim_masks
        neg_mask = ~sim_masks          # still (N, N)

        # Mean positive distance
        pos_loss = (distance_sq * pos_mask).sum() / pos_mask.sum().clamp_min(1)

        # Contrastive margin on negatives
        margin = 1.0
        neg_margin = F.relu(margin - distance_sq)        # (N, N)
        neg_loss  = (neg_margin.pow(2) * neg_mask).sum() / neg_mask.sum().clamp_min(1)

        loss = pos_loss + neg_loss

        return loss