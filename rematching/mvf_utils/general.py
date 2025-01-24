# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import torch
import logging
import einops


def get_batch_dist_matrix(a, b):
    x, y = a, b
    x_square = (x ** 2).sum(dim=-1, keepdim=True)
    y_square = (y ** 2).sum(dim=-1).unsqueeze(1)
    zz = torch.bmm(x, y.transpose(2, 1))
    P_mine = x_square + y_square - 2 * zz

    return P_mine

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_item(list,idx):
    if (len(list) > 0):
        return list[idx]
    else:
        return None

def write_to_clipboard(output):
    import subprocess
    process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode())
        
def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    if len(vectors.shape) == 2:
        vectors = vectors.unsqueeze(-1)
        squeeze = True
        squeeze_d = -1
    else:
        squeeze = False
    N, L, D = vectors.shape
    
    if indices.ndim == 1:
        squeeze = True
        squeeze_d = 1
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(squeeze_d)
    return out

def configure_logging(debug,quiet,logfile):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("equiv_shapes - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def get_cuda_ifavailable(torch_obj):
    if (torch.cuda.is_available()):
        return torch_obj.cuda()
    else:
        return torch_obj

