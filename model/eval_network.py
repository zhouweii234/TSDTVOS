"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class TSDTVOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = QueryEncoder() 
        self.memory_encoder = MemoryEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.value_proj = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()

        self.transformer = FeatureFusionNetwork()

    def encode_memory(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.memory_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_query(self, frame):
        f16, f8, f4 = self.query_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.value_proj(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16, conf_score): 
        k = mem_bank.num_objects

        readout_mem = mem_bank.match_memory(qk16, conf_score)
        qv16 = qv16.expand(k, -1, -1, -1)

        B, CV, H, W = readout_mem.shape

        hs1, hs2 = self.transformer.featurefusion_network(self.transformer.input_proj(readout_mem), self.transformer.input_proj(qv16))

        hs1 = F.interpolate(hs1, readout_mem.shape[-2:], mode='bilinear', align_corners=False)
        hs2 = F.interpolate(hs2, qv16.shape[-2:], mode='bilinear', align_corners=False)

        qv16 = torch.cat([hs1, hs2, readout_mem, qv16], 1)

        return torch.sigmoid(self.decoder(qv16, qf8, qf4))
