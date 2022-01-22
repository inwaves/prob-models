import torch
import torch.nn.functional as F
from mingpt.utils import top_k_logits


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    model.eval()

    for k in range(steps):
        logits, _ = model(x.to(model.device))

        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
