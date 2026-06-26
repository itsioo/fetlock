import torch
import torch.nn.functional as functional


def nt_xent(proj_a: torch.Tensor, proj_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = proj_a.shape[0]
    features = functional.normalize(torch.cat([proj_a, proj_b], dim=0), dim=1)
    logits = features @ features.t() / temperature
    eye = torch.eye(2 * batch, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, float("-inf"))
    targets = torch.arange(2 * batch, device=logits.device)
    targets = (targets + batch) % (2 * batch)
    return functional.cross_entropy(logits, targets)
