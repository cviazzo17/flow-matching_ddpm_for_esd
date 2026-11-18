import torch
def simulate_bernoulli_gamma(p, shape, rate, localize = 1):

    bernoulli_samples = torch.bernoulli(p)
    gamma_samples = torch.distributions.Gamma(shape, rate).sample() + localize
    
    return bernoulli_samples * gamma_samples


def simulate_bernoulli_gammaV2(p, log_shape, log_scale, localize = 1.0):

    bernoulli_samples = torch.bernoulli(p)
    shape = torch.exp(log_shape)
    rate = 1 / torch.exp(log_scale)

    gamma_samples = torch.distributions.Gamma(shape, rate).sample() + localize
    
    return bernoulli_samples * gamma_samples


