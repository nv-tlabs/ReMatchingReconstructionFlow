# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Utilities for geometry etc.
"""

import torch
from torch import Tensor
from torch import nn


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)

def sample_gaussian(mean:Tensor, logsigma:Tensor) -> Tensor:
    sample = mean + logsigma.exp() * torch.randn_like(logsigma)
    return sample

def swish(x):
    return x*torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return swish(x)

def split_colored_pc(colored_pc):
    """
    Returns:
        (xyz, color)
    """
    return colored_pc[..., :3], colored_pc[..., 3:]

def join_colored_pc(pc_xyz, pc_rgb):
    """
    Returns:
        (xyz, color)
    """
    return torch.cat([pc_xyz, pc_rgb], dim=-1)

def decomposed_L2(X,Y,W,angle_weight=0.8):
    d_l2 = {}
    angle = (W.transpose(1,2) * (2 - 2*(torch.nn.functional.normalize(X,dim=-1,p=2) * torch.nn.functional.normalize(Y,dim=-1,p=2)).sum(-1))).sum(-2).mean()

    x_norm = torch.norm(X,p=2,dim=-1)
    y_norm = torch.norm(Y,p=2,dim=-1)

    speed = (W.transpose(1,2)*(x_norm - y_norm).abs()).sum(1).mean(-1).mean()

    d_l2["rematching"] = ((angle_weight*angle + (1-angle_weight)*speed))
    d_l2["total"] = d_l2["rematching"]
    d_l2["angle"] = angle
    d_l2["speed"] = speed
    return d_l2