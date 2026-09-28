"""Phase 3 (Algorithm tab:method-4): real-world adaptation of the semantic
output only.

Only the SACR segmentation decoder (``sacr.seg_head``) is fine-tuned, with
pixel-wise cross-entropy on labelled real oil-palm images from plantations
other than the test corridor (lambda_geom = lambda_depth = 0). The encoder,
depth branch, geometry head, channel gate, CAMR, policy and AGSS keep their
simulation-trained parameters, so z_struct_aug, the actions and the metric
depth scale are unchanged; only the reported masks change.

Defaults follow the paper: 20 epochs at a learning rate of 1e-5
(``cfg.finetune_real``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.camr import CAMR
from ..models.sacr import SACR
from ..utils.logger import CSVLogger


def freeze_all_but_seg_decoder(sacr: SACR) -> list[torch.nn.Parameter]:
    for prm in sacr.parameters():
        prm.requires_grad_(False)
    params = list(sacr.seg_head.parameters())
    for prm in params:
        prm.requires_grad_(True)
    return params


def finetune_seg_decoder_real(sacr: SACR, rgb: np.ndarray, seg: np.ndarray, cfg, device,
                              logger: CSVLogger, ignore_index: int = 255) -> SACR:
    """rgb: (N, H, W, 3) uint8, seg: (N, H, W) int class ids (``ignore_index`` = unlabelled)."""
    params = freeze_all_but_seg_decoder(sacr)
    optim = torch.optim.Adam(params, lr=getattr(cfg.finetune_real, "seg_lr", 1e-5))
    epochs = getattr(cfg.finetune_real, "epochs", 20)
    batch_size = getattr(cfg.finetune_real, "batch_size", 4)
    rng = np.random.default_rng(getattr(cfg, "seed", 0))

    sacr.eval()                       # frozen parts (BatchNorm/Dropout) stay in inference mode
    sacr.seg_head.train()
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(rgb))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = torch.from_numpy(rgb[idx]).float().permute(0, 3, 1, 2).to(device) / 255.0
            y = torch.from_numpy(seg[idx]).long().to(device)
            logits = sacr(x, need_seg=True).seg_logits
            l_seg = F.cross_entropy(logits, y, ignore_index=ignore_index)

            optim.zero_grad()
            l_seg.backward()
            optim.step()

            if step % cfg.training.log_every == 0:
                logger.log(step, {"epoch": epoch, "L_seg_real": l_seg.item()})
            step += 1

    sacr.eval()
    for prm in sacr.parameters():
        prm.requires_grad_(False)
    return sacr


def finetune_real(sacr: SACR, camr: CAMR, rgb: np.ndarray, seg: np.ndarray, cfg, device, logger: CSVLogger):
    """Adapt the segmentation decoder; CAMR (and policy/AGSS) are returned unchanged."""
    sacr = finetune_seg_decoder_real(sacr, rgb, seg, cfg, device, logger)
    camr.eval()
    for prm in camr.parameters():
        prm.requires_grad_(False)
    return sacr, camr
