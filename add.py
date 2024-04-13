import numpy as np
import torch


def calculate_NCE_loss(self, src, tgt):
    n_layers = len(self.nce_layers)
    feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

    if self.opt.flip_equivariance and self.flipped_for_equivariance:
        feat_q = [torch.flip(fq, [3]) for fq in feat_q]

    feat_k = self.netG(src, self.nce_layers, encode_only=True)
    feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
    feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
        loss = crit(f_q, f_k) * self.opt.lambda_NCE
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers