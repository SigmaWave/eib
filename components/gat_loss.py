import math
import torch
import torch.nn.functional as F

FINDKG_MARGIN = 1.0

def bpr_link_loss(emb: torch.Tensor, pos_src, pos_dst, neg_src, neg_dst, margin=0.0):
    s_pos = (emb[pos_src] * emb[pos_dst]).sum(dim=1)
    s_neg = (emb[neg_src] * emb[neg_dst]).sum(dim=1)
    if s_pos.numel() == 0 or s_neg.numel() == 0: return torch.tensor(0.0, device=emb.device)
    repeat = s_neg.shape[0] // max(1, s_pos.shape[0])
    s_pos_rep = s_pos.repeat_interleave(repeat)
    return F.relu(margin + s_neg - s_pos_rep).mean()

def ce_node_loss(logits: torch.Tensor, target: torch.Tensor):
    return F.cross_entropy(logits, target)

def findkg_loss(emb: torch.Tensor, pos_src, pos_dst, neg_src, neg_dst, gamma=FINDKG_MARGIN):
    s_pos = (emb[pos_src] * emb[pos_dst]).sum(dim=1)
    s_neg = (emb[neg_src] * emb[neg_dst]).sum(dim=1)
    if s_pos.numel() == 0 or s_neg.numel() == 0: return torch.tensor(0.0, device=emb.device)
    pos_part = F.logsigmoid(s_pos - gamma)
    neg_part = F.logsigmoid(gamma - s_neg)
    repeat = s_neg.shape[0] // max(1, s_pos.shape[0])
    pos_part = pos_part.repeat_interleave(repeat)
    loss = - (pos_part + neg_part).mean()
    return loss

def info_nce_loss(emb, pos_src, pos_dst, neg_src, neg_dst, temperature=0.1):
    pos_scores = (emb[pos_src] * emb[pos_dst]).sum(dim=-1)
    neg_scores = (emb[neg_src] * emb[neg_dst]).sum(dim=-1)
    num_pos = pos_scores.shape[0]
    num_neg = neg_scores.shape[0]
    if num_pos == 0: return torch.tensor(0.0, device=emb.device)

    k = num_neg // max(1, num_pos)
    if num_neg % num_pos != 0:
         min_len = min(num_neg, num_pos * k)
         neg_scores = neg_scores[:min_len]
    neg_scores = neg_scores.view(num_pos, k)
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

    hidden_dim = emb.shape[1]
    all_scores = all_scores / math.sqrt(hidden_dim)
    all_scores = all_scores / temperature
    labels = torch.zeros(num_pos, dtype=torch.long, device=emb.device)
    return F.cross_entropy(all_scores, labels)
