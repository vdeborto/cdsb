import torch

def sample_cov(x, y=None, w=None):
    if w is None:
        w = 1 / x.shape[-2] * torch.ones((x.shape[-2], 1)).to(x.device)
    else:
        w = w.view(*x.shape[:-1], 1)
    x_centred = x - (x * w).sum(-2, keepdim=True)
    if y is None:
        y_centred = x_centred
    else:
        y_centred = y - (y * w).sum(-2, keepdim=True)
    cov = (w * x_centred).transpose(-2, -1) @ y_centred / (1 - (w**2).sum(-2, keepdim=True))  # (batch, xdim, ydim)
    return cov

def log_ess(log_w):
    ess_num = 2 * torch.logsumexp(log_w, 0)
    ess_denom = torch.logsumexp(2 * log_w, 0)
    log_ess = ess_num - ess_denom
    return log_ess

def mean_rmse(x1, x2):
    return ((((x1 - x2)**2).mean(-1))**0.5).mean()


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


if __name__ == '__main__':
    test = torch.randn(100, 5)
    assert torch.allclose(torch.cov(test.t()), sample_cov(test))

    test_1 = torch.randn(100, 5)
    test_2 = torch.randn(100, 3)
    test = torch.cat([test_1, test_2], 1)
    assert torch.allclose(torch.cov(test.t())[:5, 5:], sample_cov(test_1, test_2))