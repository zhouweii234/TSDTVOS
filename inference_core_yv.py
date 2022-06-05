import torch

from model.eval_network import TSDTVOS
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by

import math
import numpy as np

def tensor_to_numpy_SAT(t):
    r"""
    Perform naive detach / cpu / numpy process.
    :param t: torch.Tensor, (N, C, H, W)
    :return: numpy.array, (N, C, H, W)
    """
    arr = t.detach().cpu().numpy()
    return arr

class InferenceCore:
    def __init__(self, prop_net:TSDTVOS, images, num_objects, top_k=20, 
                    mem_every=5, req_frames=None, conf_thr=0.4):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.conf_thr = conf_thr

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.req_frames = req_frames

        self.top_k = top_k

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_banks = dict()
        self.scores = dict()

    def encode_query(self, idx):
        result = self.prop_net.encode_query(self.images[:,idx].cuda())
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        closest_ti = end_idx

        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_banks:
                self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k, conf_thr = self.conf_thr)
                self.scores[oi] = [1]
            self.mem_banks[oi].add_memory(key_k, key_v[i:i+1])
            if self.mem_banks[oi].get_num()!= len(self.scores[oi]):
                pred_mask = tensor_to_numpy_SAT(self.prob[oi,idx]).transpose((1, 2, 0))  #np (257,257,1)
                pred_mask_b = (pred_mask > 0.4).astype(np.uint8) #这里
                conf_score_temp = 0
                if pred_mask_b.sum() > 0:
                    conf_score_temp = (pred_mask * pred_mask_b).sum() / pred_mask_b.sum()
                else:
                    conf_score_temp = 0
                self.scores[oi].append(conf_score_temp)

        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        step = +1
        end = closest_ti - 1

        for ti in this_range: 
            is_mem_frame = (abs(ti-last_ti) >= self.mem_every)
            # Why even work on it if it is not required for memory/output
            if (not is_mem_frame) and (self.req_frames is not None) and (ti not in self.req_frames):
                continue

            k16, qv16, qf16, qf8, qf4 = self.encode_query(ti)

            # After this step all keys will have the same size
            out_mask = torch.cat([
                self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16, self.scores[oi])
            for oi in self.enabled_obj], 0)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]
                # a = self.mem_banks[oi].get_num()
            if ti != end:
                if is_mem_frame:
                    prev_value = self.prop_net.encode_memory(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    for i, oi in enumerate(self.enabled_obj):
                        self.mem_banks[oi].add_memory(prev_key, prev_value[i:i+1])
                        pred_mask = tensor_to_numpy_SAT(out_mask[i+1]).transpose((1, 2, 0))  #np (257,257,1)
                        pred_mask_b = (pred_mask > 0.4).astype(np.uint8) #这里
                        conf_score_temp = 0
                        if pred_mask_b.sum() > 0:
                            conf_score_temp = (pred_mask * pred_mask_b).sum() / pred_mask_b.sum()
                        else:
                            conf_score_temp = 0
                        self.scores[oi].append(conf_score_temp)
                    
                    if is_mem_frame:
                        last_ti = ti

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_query(frame_idx)
        key_v = self.prop_net.encode_memory(self.images[:,frame_idx].cuda(), qf16, self.prob[self.enabled_obj,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)


def softmax_w_top(x, top):
    # x = x.unsqueeze(0)
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW
    # x = x.squeeze(0)
    return x

class MemoryBank:
    def __init__(self, k, top_k=20, conf_thr=0.6):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k
        self.conf_thr=conf_thr

    def _global_matching(self, mk, qk, conf_score):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)

        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW

        T = affinity.shape[-2]//affinity.shape[-1]
        HW = affinity.shape[-1]

        for i in range(T):
            # a = affinity[j,i*HW:(i+1)*HW+1]
            if conf_score[i] < self.conf_thr: 
                affinity[:,i*HW:(i+1)*HW] = affinity[:,i*HW:(i+1)*HW]*conf_score[i]

        # if conf_score != -1:
        #     T = affinity.shape[-2]//affinity.shape[-1]
        #     HW = affinity.shape[-1]
        #     BScore = len(conf_score[0])
        #     affinity = affinity.expand(BScore,-1,-1)
        #     for i in range(T):
        #         for j in range(BScore):
        #             # a = affinity[j,i*HW:(i+1)*HW+1]
        #             if conf_score[i][j] < 0.8: 
        #                 affinity[j,i*HW:(i+1)*HW+1] = affinity[j,i*HW:(i+1)*HW+1]*conf_score[i][j]
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk, conf_score):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        mk = self.mem_k
        mv = self.mem_v

        affinity = self._global_matching(mk, qk, conf_score)

        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def get_num(self):
        return self.num

    def add_memory(self, key, value):
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.len = key.shape[2]
            self.num = 1
        else:
            # maxlen = 2
            # if self.mem_k.shape[2] >= self.len*maxlen:
            #     self.mem_k = torch.cat([self.mem_k[:,:,:self.len], key], 2)
            #     self.mem_v = torch.cat([self.mem_v[:,:,:self.len], value], 2)
            # else:
            #     self.mem_k = torch.cat([self.mem_k, key], 2)
            #     self.mem_v = torch.cat([self.mem_v, value], 2)
            self.mem_k = torch.cat([self.mem_k, key], 2)
            self.mem_v = torch.cat([self.mem_v, value], 2)
            self.num += 1
            # self.mem_k = key
            # self.mem_v = value
