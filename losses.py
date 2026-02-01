import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# in SGVB we calculate ELBO without the expectation p(x|z) term. Maximizing the p(x|z) term is equivalent to minimziing MSE later.
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(2.0 * torch.tensor(torch.pi)).to(device)
    log_likelihood = (-1/2)*(log2pi + logvar + (sample-mean)**2/torch.exp(logvar))
    return torch.sum(log_likelihood, dim=raxis)

def loss_SGVB(output):
    torch_zero = torch.tensor(0.0).to(device)
    logpz = log_normal_pdf(output['z'], torch_zero, torch_zero)
    logqz_x = log_normal_pdf(output['z'], output['mean'], output['logvar'])
    return logpz -logqz_x

def loss_KL_wo_E(output):
    var = torch.exp(output['logvar'])
    logvar = output['logvar']
    mean = output['mean']

    return -0.5 * torch.sum(torch.pow(mean, 2)
                            + var - 1.0 - logvar,
                            dim=[1])

def loss_func(output, x, coeff=1e-3):
    mse = torch.nn.MSELoss(reduction='none')
    analytical_KL = loss_KL_wo_E(output)
    err = mse(output['imgs'], x)
    err = torch.mean(err, dim=[1,2,3])
    elbo = err - coeff * analytical_KL
    return torch.mean(elbo)