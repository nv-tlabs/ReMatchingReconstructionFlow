# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

def full_name(short_name):
    if short_name == "P1":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_P1"
    elif short_name == "P3":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P3"
    elif short_name == "P4":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P4"
    elif short_name == "P1_P3":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P1_P3"
    elif short_name == "P1_P4":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P1_P4"
    elif short_name == "P3_Image":
        return "rematching.rematching_loss.FunctionReMatchingLoss_Image_P3"
    else:
        print("Invalid prior name: "+short_name)
