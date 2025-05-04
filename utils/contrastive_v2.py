from typing import Optional
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F
from utils.contrastive import ContrastManager
from scene.cameras import Camera

from scene.dataset_readers import SceneType


class ContrastManagerV2(ContrastManager):
    def __init__(
        self, 
        cfg: DictConfig, 
        example_cam: Camera, # A training camera 
        valid_mask_by_name: Optional[dict],
        scene_type: SceneType,
    ) -> None:
        super().__init__(cfg, example_cam, valid_mask_by_name, scene_type)

        print("Initializing ContrastiveManagerV2...")

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
        if instance_labels.dim() == 1: # (N, ), integer labels
            sim_masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone()) # (N, N)
            sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)
        elif instance_labels.dim() == 2: # (N, L), multi-hot labels, in bool type
            inter = instance_labels.unsqueeze(1) & instance_labels.unsqueeze(0) # (N, N, L) 
            union = instance_labels.unsqueeze(1) | instance_labels.unsqueeze(0) # (N, N, L)
            sim_masks = inter.float().sum(dim=-1) / union.float().sum(dim=-1) # (N, N)
            sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)
            sim_masks = sim_masks ** sim_exp # (N, N)
        else:
            raise Exception(f"instance_labels.dim() {instance_labels.dim()} is not supported")

        # compute similarity matrix based on Euclidean distance
        distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1) # (N, N)

        # temperature = 1 for positive pairs and temperature (100) for negative pairs
        temperature = torch.ones_like(distance_sq) * temperature # (N, N)
        temperature = torch.where(sim_masks==1, temperature, torch.ones_like(temperature)) # (N, N)

        # Process and apply the weights
        if weight is not None:
            weight_matrix = weight.unsqueeze(1) * weight.unsqueeze(0) # (N, N)

        similarity_kernel = torch.exp(-distance_sq/temperature) # (N, N)
        if weight is not None and weighting_mode == "on_sim":
            similarity_kernel = similarity_kernel * weight_matrix # (N, N)
            
        # V2: remove the redundant exp! (A recommendation from ChatGPT o3)
        # prob_before_norm = torch.exp(similarity_kernel) # (N, N)
        prob_before_norm = similarity_kernel
        if weight is not None and weighting_mode == "on_prob":
            prob_before_norm = prob_before_norm * weight_matrix # (N, N)

        if sum_in_log: 
            # First sum over positive pairs and then log - better in handling noises. 
            Z = prob_before_norm.sum(dim=-1) # (N,), denom
            p = torch.mul(prob_before_norm, sim_masks).sum(dim=-1) # (N,), numer
            prob = torch.div(p, Z) # (N,)
            prob_masked = torch.masked_select(prob, prob.ne(0)) # (N,)
            loss = - prob_masked.log().sum()/bsize # (1,)
        else:
            # First take the log and then sum over positive pairs - forcing more precise matching. 
            Z = prob_before_norm.sum(dim=-1, keepdim=True) # (N, N), denom
            prob = torch.div(prob_before_norm, Z) # (N, N)
            log_prob = torch.log(prob) # (N, N)

            # Get only the positive pairs (with similarity larger than 0)
            weighted_log_prob = torch.mul(log_prob, sim_masks) # (N, N)
            # Normalized by the number of positive pairs for each anchor
            weighted_log_prob = weighted_log_prob / (sim_masks.ne(0).sum(-1, keepdim=True) + 1e-6) # (N, N)
            # Sum over positive pairs
            log_prob_masked = torch.masked_select(weighted_log_prob, weighted_log_prob.ne(0)) # (N,)
            # Sum over anchors and normalized by the batch size
            loss = - log_prob_masked.sum()/bsize # (1, )

        return loss