# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch.nn as nn
import torch
import numpy as np
import math
import torch.autograd.forward_ad as fwAD
from .mvf_utils.embedding import get_embedder
from .mvf_utils.math_utils_torch import decomposed_L2


class Matching():
    def matching_P4(gamma, gamma_dot, W):
        """ Analytic matching procedure for the piecewise rigid deformation prior (equation 23), batched input support (b=batch size)

        Args:
            gamma (torch.Tensor): size [b, n, 3]
            gamma_dot (torch.Tensor): size [b, n, 3]
            W (torch.Tensor): size [b, n, k]

        Returns:
            torch.Tensor: size [b, k, 3, n], element from prior class P4 that is closest to the input reconstruction flow
        """        
        W = W + 1e-10

        gamma = (gamma.unsqueeze(-2)).permute(0,2,1,3)
        gamma_dot = gamma_dot

        cov = torch.matmul(gamma.transpose(2,3),W.transpose(1,2).unsqueeze(-1)*gamma)
        eye_3 = torch.eye(3).to(cov).unsqueeze(0).tile(1,1,1)

        A_block = torch.kron(eye_3,cov) + torch.kron(cov,eye_3)
        B_block = (torch.kron(torch.matmul(gamma.permute(0,1,3,2),W.permute(2,1,0))[:,:,:,0:1],eye_3) - torch.kron(eye_3,torch.matmul(gamma.permute(0,1,3,2),W.permute(2,1,0))[:,:,:,0:1]))
        C_block = -(1./W.sum(1)).unsqueeze(-1).unsqueeze(-1)*torch.kron(eye_3,torch.matmul(W.transpose(1,2).unsqueeze(-2), gamma))

        D = torch.tensor([[0,0,0],
                            [-1,0,0],
                            [0,-1,0],
                            [1,0,0],
                            [0,0,0],
                            [0,0,-1],
                            [0,1,0],
                            [0,0,1],
                            [0,0,0]]).to(A_block).unsqueeze(0).tile(gamma.shape[0],1,1,1)

        L = torch.cat([torch.eye(9)[3:4,:],torch.eye(9)[6:7,:],torch.eye(9)[7:8,:]],dim=0).to(D).unsqueeze(0).tile(gamma.shape[0],1,1,1)

        A_block_inv = torch.matmul(L,torch.linalg.inv((A_block - torch.matmul(B_block,C_block))))
        B_block_inv = -torch.matmul(A_block_inv,B_block)
        C_block_inv = -torch.matmul(C_block,torch.linalg.inv(A_block - torch.matmul(B_block,C_block)))
        D_block_inv = eye_3 - torch.matmul(C_block_inv,B_block)

        conc = torch.cat([torch.cat([A_block_inv,B_block_inv],dim=-1),torch.cat([C_block_inv,D_block_inv],dim=-1)],dim=2)

        rhs = -torch.matmul(gamma.transpose(2,3),(W.transpose(1,2).unsqueeze(-1) * gamma_dot.unsqueeze(1))) + torch.matmul(gamma_dot.transpose(1,2).unsqueeze(1),W.transpose(1,2).unsqueeze(-1)*gamma)

        sol = torch.matmul(conc,
                    torch.cat([torch.matmul(D,torch.cat([rhs[:,:,0,1].unsqueeze(-1),
                                                        rhs[:,:,0,2].unsqueeze(-1),
                                                        rhs[:,:,1,2].unsqueeze(-1)],dim=-1).unsqueeze(-1)),
                                (torch.matmul(W.transpose(1,2),gamma_dot)/W.sum(1).unsqueeze(-1)).unsqueeze(-1)],dim=2))

        A = torch.cat([torch.cat([torch.zeros_like(sol[:,:,1:2]),sol[:,:,0:1],sol[:,:,1:2]],dim=-1),
                            torch.cat([-sol[:,:,0:1],torch.zeros_like(sol[:,:,1:2]),sol[:,:,2:3]],dim=-1),
                            torch.cat([-sol[:,:,1:2],-sol[:,:,2:3],torch.zeros_like(sol[:,:,1:2])],dim=-1)],dim=2).detach()
        b = sol[:,:,3:].detach()
        A_b_recon = torch.matmul(A,gamma.transpose(2,3)) + b

        return A_b_recon.transpose(2,3)

    def div_free_basis(x, k):
        """ Create divergence free basis (equation 20), batched input support (b=batch size)
        """ based on https://github.com/marvin-eisenberger/hamiltonian-interpolation (Eisenberger et al., 2019)

        Args:
            x (torch.Tensor): size [b, n, 3]
            k (int)

        Returns:
            torch.Tensor: divergence free basis
        """        
        kv = torch.arange(1, k+1, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2).cuda()
        x = x.unsqueeze(3) * kv * math.pi

        vert_sin = torch.sin(x)
        vert_cos = torch.cos(x) * kv

        if x.shape[2] == 2: 
            hat_matrix = torch.as_tensor([[[1, 1], [-1, 1]]], dtype=torch.float32).cuda()
      
            sin_x = vert_sin[:, :, 0, :].unsqueeze(3).unsqueeze(4)
            sin_y = vert_sin[:, :, 1, :].unsqueeze(2).unsqueeze(4)

            cos_x = vert_cos[:, :, 0, :].unsqueeze(3).unsqueeze(4)
            cos_y = vert_cos[:, :, 1, :].unsqueeze(2).unsqueeze(4)

            u = torch.cat(((cos_x * sin_y).unsqueeze(2),
                            (sin_x * cos_y).unsqueeze(2),
                            ), 2)

            scale_fac = torch.sqrt(kv.unsqueeze(3) ** 2 + kv.unsqueeze(4) ** 2) ** (-1)

            scale_fac = scale_fac.transpose(2, 4).unsqueeze(5)

            scale_fac = torch.cat((scale_fac.unsqueeze(2).repeat_interleave(k, 2),
                                scale_fac.unsqueeze(3).repeat_interleave(k, 3),
                                ), 6)
          
        else:
            hat_matrix = torch.as_tensor([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                                [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                                [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]], dtype=torch.float32).cuda()
        
            sin_x = vert_sin[:, :, 0, :].unsqueeze(3).unsqueeze(4)
            sin_y = vert_sin[:, :, 1, :].unsqueeze(2).unsqueeze(4)
            sin_z = vert_sin[:, :, 2, :].unsqueeze(2).unsqueeze(3)

            cos_x = vert_cos[:, :, 0, :].unsqueeze(3).unsqueeze(4)
            cos_y = vert_cos[:, :, 1, :].unsqueeze(2).unsqueeze(4)
            cos_z = vert_cos[:, :, 2, :].unsqueeze(2).unsqueeze(3)

            u = torch.cat(((cos_x * sin_y * sin_z).unsqueeze(2),
                            (sin_x * cos_y * sin_z).unsqueeze(2),
                            (sin_x * sin_y * cos_z).unsqueeze(2)), 2)

            scale_fac = torch.sqrt(kv.unsqueeze(3) ** 2 + kv.unsqueeze(4) ** 2) ** (-1)

            scale_fac = scale_fac.transpose(2, 4).unsqueeze(5)

            scale_fac = torch.cat((scale_fac.unsqueeze(2).repeat_interleave(k, 2),
                                scale_fac.unsqueeze(3).repeat_interleave(k, 3),
                                scale_fac.unsqueeze(4).repeat_interleave(k, 4)), 6)

        u = u.transpose(2, 5).unsqueeze(6).unsqueeze(7)

        u = torch.sum(hat_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) * u, 5)
        u = u * scale_fac

        u = u.transpose(2, 5).reshape(x.shape[0], x.shape[1], x.shape[2], -1).transpose(2, 3)

        return u

    def matching_combination_P3(gamma, gamma_dot, W, B):
        """ Analytic matching procedure for the adaptive combination prior made up of multiple P3 prior classes (equation 19, 21), batched input support (b=batch size)

        Args:
            gamma (torch.Tensor): size [b, n, 3]
            gamma_dot (torch.Tensor): size [b, n, 3]
            W (torch.Tensor): size [b, n, k]
            B (int): divergence-free basis dimensionality

        Returns:
            torch.Tensor: element from each prior class P3 that is closest to the input reconstruction flow (weighted by W)
        """   
               
        u = Matching.div_free_basis(gamma,B)

        beta_sol = torch.matmul(torch.matmul(gamma_dot.unsqueeze(2).unsqueeze(1),(W.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * u.unsqueeze(1)).transpose(3,4)).mean(2),
                        torch.linalg.pinv(torch.matmul((1e-10+W.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * u.unsqueeze(1)),(1e-10+W.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * u.unsqueeze(1)).transpose(3,4)).mean(2),hermitian=True))
        
        beta_recon = (beta_sol.unsqueeze(-1)*(u.unsqueeze(1))).sum(3)

        return beta_recon

    def matching_P3(X,G,S,B):
        """ Analytic matching procedure for the volume preserving prior (equation 19, 22)

        Args:
            X (torch.Tensor): size [b, n, 2]
            G (torch.Tensor): size [b, n, 2] gradient of psi 
            S (torch.Tensor): size [b, n], derivative of psi by time
            B (int): divergence-free basis dimensionality

        Returns:
            torch.Tensor: size [b, n, 1, 2], element from each prior class P3 that is closest to the input reconstruction flow (weighted by W)
        """   

        u = Matching.div_free_basis(X,B)
        
        B_sol = S
        A_sol = (u*G.unsqueeze(2)).sum(-1)
        beta_sol = torch.linalg.lstsq(A_sol,-B_sol)[0].detach()

        beta_recon = (beta_sol.unsqueeze(1).unsqueeze(-1)*u).sum(2).unsqueeze(-2)

        return beta_recon
    
class ReMatching(nn.Module):
    def __init__(self,
                **kwargs) -> None:
        super().__init__()
        general = kwargs["general"]
        prior = kwargs["prior"]

        self.t_multires = prior.get("adaptive_prior.t_multires",6)
        self.W_hidden_dim = prior.get("adaptive_prior.W_hidden_dim",256)
        self.projected_weight = prior.get("P1.projected_weight",0.3)
        self.rm_weight = general.get("rm_weight",0.001)
        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.K = prior.get("adaptive_prior.K",5)
        self.entropy_weight = prior.get("adaptive_prior.entropy_weight",0.0001)
        self.B = prior.get("P3.B",1)
        self.activation = torch.nn.Softplus(beta=100)
        self.V = torch.tensor(prior.get("P1.V",[[0.0,0,1]])).cuda() 

        if self.K>0:
            self.W_net = nn.ModuleList([nn.Linear(3+3+time_input_ch, self.W_hidden_dim), nn.Linear(self.W_hidden_dim,self.K)]).cuda()

    def train_setting(self, training_args):
        self.spatial_lr_scale = 5
        l = [{'params': list(self.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "rematching"}]
        # This optimizer is added to make the rematching a standalone class, but it can be removed and the parameters 
        # can be added to the base reconstruction network optimizer
        self.optimizer = torch.optim.Adam(l, lr=0, eps=1e-15)

        from utils.general_utils import get_expon_lr_func

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def rematching_loss(self, psi, reg=1):
        """ This is a function for caculating the ReMatching loss on discrete sets

        Args:
            psi (tuple): set of parameters determining the current reconstruction flow (last parameter: scene decomposition weights, can be set to None for built-in computation)

        Returns:
            dict: dictionary containing the speed/angle decomposition of the ReMatching loss, the total 
                ReMatching loss and the scene decomposition weights
        """     
        # Scene decomposition
        W = psi[-1]
        W_all_unnorm = None
        if W == None:
            W_all_unnorm = self.get_W_unnorm(psi)
            W = torch.nn.functional.softmax(W_all_unnorm,dim=2) 
            psi = psi[:-1] + (W,W_all_unnorm)
        else:
            psi = psi + (W,)

        # Matching
        prior_recon = self.match(psi)

        # Rematching
        rematching_loss = self.rematch(psi, prior_recon)

        rematching_loss["total"] *= self.rm_weight
        rematching_loss["total"] += self.entropy_weight * rematching_loss["entropy"]
        rematching_loss["total"] /= reg

        return rematching_loss

class DiscreteReMatchingLoss(ReMatching):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def forward(self, gaussians, deformation, time, rendering_package=None, W=None):
        """ This is a function that conducts a forward deformation step and returns derivatives necessary for the ReMatching procedure

        Args:
            gaussians (scene.gaussian_model.GaussianModel): canonical Gaussians
            deformation (scene.deform_model.DeformModel): Gaussian time-dependent deformation
            time (torch.Tensor): randomly sampled time for the evaluation of ReMatching
            rendering_package (tuple): rendering parameters and functions
            W (torch.Tensor): precomputed W
        """        
        xyz = gaussians.get_xyz

        with fwAD.dual_level():
            dual_input = fwAD.make_dual(time.clone(), torch.ones_like(time))
            deformed_output, _, _ = deformation.step(gaussians.get_xyz.detach(),dual_input)
            unpacked = fwAD.unpack_dual(deformed_output)
            d_xyz_der = unpacked.tangent
            d_xyz = unpacked.primal

        return (xyz.unsqueeze(0),d_xyz.unsqueeze(0),d_xyz_der.unsqueeze(0),time.unsqueeze(0),W),1
    
    def rematch(self, psi, prior_recon):
        xyz, d_xyz, gamma_dot, time, W, W_all_unnorm = psi
        gamma = xyz + d_xyz

        rematching_loss = decomposed_L2(prior_recon,gamma_dot,W)

        rematching_loss["W"] = (W,W_all_unnorm)

        # Entropy loss calculation
        # Ensure that this is the top call of the W prediction
        if self.entropy_weight>0 and W_all_unnorm!=None: 
            rematching_loss["entropy"] = self.entropy_loss(gamma.detach(),W,W_all_unnorm)
        else:
            rematching_loss["entropy"] = torch.tensor(0).cuda()
    
        return rematching_loss
    
    def entropy_loss(self, gamma, W, W_all_unnorm):
        """ This is a function that computes a scaled entropy loss, where the scaling is determined by 
        the sizes of bounding boxes constraining all gaussians belonging to a specific class

        Args:
            gamma (torch.Tensor): gaussian center coordinates at chosen time
            W (torch.Tensor): class probabilities for all Gaussians
            W_all_unnorm (torch.Tensor): unnormalized class probabilities for all Gaussians

        Returns:
            torch.Tensor: entropy loss
        """        
        current_classes = torch.argmax(W,dim=2)
        
        xs = torch.zeros((W_all_unnorm.shape[2],1)).cuda()
        ys = torch.zeros((W_all_unnorm.shape[2],1)).cuda()
        zs = torch.zeros((W_all_unnorm.shape[2],1)).cuda()

        for i in range(W_all_unnorm.shape[2]):
            selected = gamma[current_classes==i]
            if selected.shape[0]>3:
                _,S,V = torch.pca_lowrank(selected, q=None, center=True, niter=2)
                selected_oriented = torch.matmul(selected, V[:, :3])
                xs[i,0] = selected_oriented[:,0].max() - selected_oriented[:,0].min()
                ys[i,0] = selected_oriented[:,1].max() - selected_oriented[:,1].min()
                zs[i,0] = selected_oriented[:,2].max() - selected_oriented[:,2].min()

        S = (xs*ys + xs*zs + ys*zs + 1e-2).squeeze(1).detach()
        S = (1/S)
        S = S/S.sum()
        n = torch.tensor(W_all_unnorm.shape[1])
        log_W = torch.logsumexp(W_all_unnorm - torch.logsumexp(W_all_unnorm,dim=2).unsqueeze(-1),dim=1) - torch.log(n)

        return (S * (1/n) * W.sum(dim=1) * log_W).sum()
    
    def get_W_unnorm(self, psi):
        xyz, d_xyz, _, time, _ = psi
        gamma = xyz + d_xyz

        if self.K > 0:
            W_input = torch.cat([gamma.detach(),xyz.detach(),self.embed_time_fn(time)],dim=2)
            for i, _ in enumerate(self.W_net):
                W_input = self.W_net[i](W_input)
                if i<1:
                    W_input = self.activation(W_input)
            W_all_unnorm = W_input
        else:
            W_all_unnorm = torch.ones((xyz.shape[0],1)).cuda()

        return W_all_unnorm
    
    def render_W_decomposition(self, camera, time, gaussians, deform, rendering_package):
        torch.manual_seed(0)
        import distinctipy
        colors = torch.cat((torch.tensor([[1,0,0]]),torch.tensor([[0,1,0]]),torch.tensor([[0,0,1]]),torch.tensor(distinctipy.get_colors(self.K-3)))).cuda()

        render, pipe, background = rendering_package
        xyz = gaussians.get_xyz
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time)

        store_features_dc = gaussians._features_dc
        store_features_rest = gaussians._features_rest
        psi = xyz.unsqueeze(0),d_xyz.unsqueeze(0), None, time.unsqueeze(0), None
        components = torch.argmax(self.get_W_unnorm(psi),dim=2)[0]
        comp_colors = torch.zeros_like(gaussians._features_dc)
        
        for k in range(colors.shape[0]):
            comp_colors += (components == k).unsqueeze(-1).unsqueeze(-1) * colors[k].unsqueeze(0)
        gaussians._features_dc = comp_colors
        gaussians._features_rest = torch.zeros_like(gaussians._features_rest)

        render_pkg_re = render(camera, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, False)
        image, _, _, _ = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        gaussians._features_dc = store_features_dc
        gaussians._features_rest = store_features_rest


        return image

class DiscreteReMatchingLoss_P1(DiscreteReMatchingLoss):
    # Directional restricted class (equation 11)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def match(self, psi):
        gamma_dot = torch.nn.functional.normalize(psi[2],dim=-1).float()
        return (gamma_dot - (torch.matmul(gamma_dot,self.V.T) * self.V.T.cuda().unsqueeze(1)).sum(2).T)
    
    def rematch(self, psi, prior_recon):
        W = psi[-2].squeeze(-1)
        gamma_dot = torch.nn.functional.normalize(psi[2],dim=-1).float()
        rematching_loss = {}
        rematching_loss["total"] = (W.unsqueeze(0)*(torch.linalg.norm(gamma_dot - prior_recon,dim=-1))).mean()
        rematching_loss["entropy"] = torch.tensor(0).cuda()
        rematching_loss["speed"] = torch.tensor(0).cuda()
        rematching_loss["angle"] = torch.tensor(0).cuda()
        return rematching_loss

class DiscreteReMatchingLoss_AdaptivePriors_P4(DiscreteReMatchingLoss):
    # Piecewise rigid motion class (equation 23)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def match(self, psi):
        xyz, d_xyz, gamma_dot, _, W, _ = psi
        gamma = xyz + d_xyz
        return Matching.matching_P4(gamma.detach(), gamma_dot.detach(), W.detach())
    
class DiscreteReMatchingLoss_AdaptivePriors_P3(DiscreteReMatchingLoss):
    # Set of volume preserving priors
    def __init__(self, **kwargs) -> None:
        if "B" not in kwargs.keys():
            kwargs["B"] = 1
        super().__init__(**kwargs)

    def match(self, psi):
        xyz, d_xyz, gamma_dot, time, W, _ = psi
        gamma = xyz + d_xyz
        return Matching.matching_combination_P3(gamma.detach(), gamma_dot.detach(), W.detach(), self.B)
     
class DiscreteReMatchingLoss_AdaptivePriors_P1_P4(DiscreteReMatchingLoss):
    # Piecewise rigid motion class with a single directional restricted class
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.p4 = DiscreteReMatchingLoss_AdaptivePriors_P4(**kwargs)
        self.p1 = DiscreteReMatchingLoss_P1(**kwargs)
    
    def match(self, psi):
        xyz, d_xyz, gamma_dot, time, W, W_all_unnorm = psi
        gamma = xyz + d_xyz

        if W == None:
            W_all_unnorm = self.get_W_unnorm(psi)
            W = torch.nn.functional.softmax(W_all_unnorm, dim=1)
        
        prior_recon_p1 = self.p1.match((xyz, d_xyz, gamma_dot, time, W[:,:,:1], W_all_unnorm[:,:,:1]))
        prior_recon_p4 = self.p4.match((xyz, d_xyz, gamma_dot, time, W[:,:,1:], W_all_unnorm[:,:,1:]))
        

        return (prior_recon_p4,prior_recon_p1)

    def rematch(self, psi, prior_recon):
        xyz, d_xyz, gamma_dot, time, W, W_all_unnorm = psi
        gamma = xyz + d_xyz

        prior_recon_p4, prior_recon_p1 = prior_recon

        rematching_loss_p1 = self.p1.rematch((xyz, d_xyz, gamma_dot, time, W[:,:,:1], W_all_unnorm[:,:,:1]),prior_recon_p1)
        rematching_loss_p4 = self.p4.rematch((xyz, d_xyz, gamma_dot, time, W[:,:,1:], W_all_unnorm[:,:,1:]),prior_recon_p4)

        rematching_loss = {}
        rematching_loss["W"] = (W, W_all_unnorm)

        rematching_loss["projected"] = rematching_loss_p1["total"]
        rematching_loss["angle"] = rematching_loss_p4["angle"]
        rematching_loss["speed"] = rematching_loss_p4["speed"]
        rematching_loss["total"] = self.projected_weight*rematching_loss_p1["total"] + (1-self.projected_weight)*rematching_loss_p4["total"]

        if self.entropy_weight>0 and W_all_unnorm!=None: # Ensure that this is the top call of the W prediction
            rematching_loss["entropy"] = self.entropy_loss(gamma.detach(),W,W_all_unnorm)
        else:
            rematching_loss["entropy"] = torch.tensor(0).cuda()

        return rematching_loss
    
class DiscreteReMatchingLoss_AdaptivePriors_P1_P3(DiscreteReMatchingLoss):
    # Piecewise volume preserving class with a single directional restricted class
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.p3 = DiscreteReMatchingLoss_AdaptivePriors_P3(**kwargs)
        self.p1 = DiscreteReMatchingLoss_P1(**kwargs)
    
    def match(self, psi):
        xyz, d_xyz, gamma_dot, time, W, W_all_unnorm = psi
        gamma = xyz + d_xyz

        if W == None:
            W_all_unnorm = self.get_W_unnorm(psi)
            W = torch.nn.functional.softmax(W_all_unnorm, dim=1)

        prior_recon_p3 = self.p3.match((xyz, d_xyz, gamma_dot, time, W[:,:,1:], W_all_unnorm[:,:,1:]))
        prior_recon_p1 = self.p1.match((xyz, d_xyz, gamma_dot, time, W[:,:,:1], W_all_unnorm[:,:,:1]))

        return (prior_recon_p3,prior_recon_p1)
    
    def rematch(self, psi, prior_recon):
        xyz, d_xyz, gamma_dot, time, W, W_all_unnorm = psi
        gamma = xyz + d_xyz

        prior_recon_p3, prior_recon_p1 = prior_recon

        rematching_loss_p1 = self.p1.rematch((xyz, d_xyz, gamma_dot, time, W[:,:,:1], W_all_unnorm[:,:,:1]),prior_recon_p1)
        rematching_loss_p3 = self.p3.rematch((xyz, d_xyz, gamma_dot, time, W[:,:,1:], W_all_unnorm[:,:,1:]),prior_recon_p3)

        rematching_loss = {}
        rematching_loss["W"] = (W, W_all_unnorm)

        rematching_loss["projected"] = rematching_loss_p1["total"]
        rematching_loss["angle"] = rematching_loss_p3["angle"]
        rematching_loss["speed"] = rematching_loss_p3["speed"]
        rematching_loss["total"] = self.projected_weight*rematching_loss_p1["total"] + (1-self.projected_weight)*rematching_loss_p3["total"]

        if self.entropy_weight>0 and W_all_unnorm!=None: # Ensure that this is the top call of the W prediction
            rematching_loss["entropy"] = self.entropy_loss(gamma.detach(),W,W_all_unnorm)
        else:
            rematching_loss["entropy"] = torch.tensor(0).cuda()

        return rematching_loss

class FunctionReMatchingLoss(ReMatching):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class FunctionReMatchingLoss_Image(FunctionReMatchingLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        scene = kwargs["scene"]
        self.cam_time = kwargs["prior"].cam_time
        viewpoint_stack = scene.getTrainCameras().copy()

        # Camera selection for flow evaluation
        if self.cam_time>0:
            self.fix_viewpoint_camera = list(filter(lambda x: torch.abs(x.fid-torch.tensor(self.cam_time))<1e-7,viewpoint_stack))[0]
        else:
            self.fix_viewpoint_camera = viewpoint_stack[0]


    def forward(self, gaussians, deformation, time, rendering_package):
        """ This is a function that conducts a forward deformation step and returns derivatives necessary for the ReMatching procedure

        Args:
            gaussians (scene.gaussian_model.GaussianModel): canonical Gaussians
            deformation (scene.deform_model.DeformModel): Gaussian time-dependent deformation
            time (torch.Tensor): randomly sampled time for the evaluation of ReMatching
            rendering_package (tuple): rendering parameters and functions
        """        
        from torchvision import transforms

        self.fix_viewpoint_camera.image_width = 160
        self.fix_viewpoint_camera.image_height = 160
        self.fix_viewpoint_camera.focal_x = self.fix_viewpoint_camera.image_width/np.tan(self.fix_viewpoint_camera.FoVx)
        self.fix_viewpoint_camera.focal_y = self.fix_viewpoint_camera.image_height/np.tan(self.fix_viewpoint_camera.FoVy)
        render, pipe, background = rendering_package

        xyz = gaussians.get_xyz 
    
        def render_zeroth(time):
            alpha = 0.1
            d_xyz, d_rotation, d_scaling = deformation.step(xyz.detach(), time)

            # Insert a pytorch version of 3D Gaussian Renderer that supports the use of the forward-mode derivative calculation
            # Since such a renderer was not available as licensed open source software at the time of our code release, we leave 
            # the selection and setup of the appropriate renderer to the user (store renderer in g)
            g = None
            if g == None:
                print("To use the Function ReMatching loss on images, please add renderer in rematching loss file.")

            gaussians._xyz = gaussians._xyz + d_xyz
            gaussians._rotation = gaussians._rotation + d_rotation
            gaussians._scaling = gaussians._scaling + d_scaling
            fix_image = g(self.fix_viewpoint_camera,gaussians)["render"].clip(0,1)
            gray_image = 2*transforms.functional.rgb_to_grayscale(fix_image.permute(2,0,1)) - 1

            transformed_image = -(alpha * torch.log(1 - torch.abs(gray_image) + 1e-10) * torch.sign(gray_image)).permute(1,2,0)

            mask = (torch.cat([torch.abs(fix_image)],dim=-1)>1e-2).any(dim=2).flatten()
            
            gaussians._xyz = gaussians._xyz - d_xyz
            gaussians._rotation = gaussians._rotation - d_rotation
            gaussians._scaling = gaussians._scaling - d_scaling
            return mask, transformed_image

        with fwAD.dual_level():
            fid_input = fwAD.make_dual(time.requires_grad_(True), torch.ones_like(time))
            mask, fix_image = render_zeroth(fid_input)
            fix_image_der = fwAD.unpack_dual(fix_image).tangent
           
        self.fix_viewpoint_camera.image_width = 800
        self.fix_viewpoint_camera.image_height = 800

        grad_x = torch.cat((torch.zeros_like(fix_image[:1,:,:]),-fix_image[:-2,:,:] + fix_image[2:,:,:],torch.zeros_like(fix_image[:1,:,:])),dim=0)
        grad_y = torch.cat((torch.zeros_like(fix_image[:,:1,:]),-fix_image[:,:-2,:] + fix_image[:,2:,:],torch.zeros_like(fix_image[:,:1,:])),dim=1)

        Gs = torch.stack((grad_x,grad_y),dim=0).flatten(start_dim=1,end_dim=-2).permute(2,1,0)
        
        Xs = torch.stack((torch.arange(0,fix_image.shape[0]).unsqueeze(-1).tile((1,fix_image.shape[1])),torch.arange(0,fix_image.shape[1]).tile((fix_image.shape[0],1)))).flatten(start_dim=1,end_dim=-1).T.unsqueeze(0).float().cuda()
        X_norms = Xs-Xs.mean()
        X_norms = X_norms/(X_norms.max())
        Ts = fix_image_der.flatten(start_dim=0,end_dim=-2).T

        W = torch.ones_like(Ts[0,mask])
    
        return (X_norms[:,mask],Gs[:,mask],Ts[:,mask],2,W),fix_image.shape[2]

class FunctionReMatchingLoss_Image_P3(FunctionReMatchingLoss_Image):
    # Image space ReMatching loss (section 8.2.3)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def match(self, psi):
        X,G,S,B,_,_ = psi
        return Matching.matching_P3(X,G,S,B)
        
    def rematch(self, psi, beta_recon):
        _,G,S,_,_,_= psi
        rematching_loss = {}
        rematching_loss["total"] = torch.abs(torch.matmul(beta_recon,G.unsqueeze(-1)).squeeze(-1).squeeze(-1) + S).mean()
        return rematching_loss
