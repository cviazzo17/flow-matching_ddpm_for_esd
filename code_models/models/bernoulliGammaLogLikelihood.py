import torch
import torch.nn as nn
 

# fixme clamp avoid
# fixme use double precision ?

class negativeBernoulliGammaLogLikelihood(nn.Module):
# p, alpha, beta. Structure:(N, :) 
# observed. Structure:(N, :)


    def __init__(self, wet_day_threshold = 1, epsilon_if_zero = 0.00001, localize = True):
        super().__init__()
        self.wet_day_threshold = wet_day_threshold
        self.epsilon_if_zero = epsilon_if_zero
        
        if localize:
            self.locate = wet_day_threshold
        else:
            self.locate = 0
        
        
        
    def forward(self, observed, predicted):
        
        p, alpha, beta = predicted[:, [0], ...], predicted[:, [1], ...], predicted[:, [2], ...]
               
        positiveIndex = (observed > self.wet_day_threshold)
        nullIndex = (observed <= self.wet_day_threshold)

        ll =  torch.sum( torch.log( torch.clamp(p[positiveIndex], min = self.epsilon_if_zero) ) + \
                        (alpha[positiveIndex]-1)*torch.log( observed[positiveIndex] - self.locate ) - \
                        (observed[positiveIndex] - self.locate)*beta[positiveIndex] + \
                alpha[positiveIndex]*torch.log( torch.clamp(beta[positiveIndex], min = self.epsilon_if_zero) )  - \
                            torch.lgamma( torch.clamp(alpha[positiveIndex], min = self.epsilon_if_zero) ) 
                        ) + torch.sum( torch.log( 1 - torch.clamp(p[nullIndex], max = (1 - self.epsilon_if_zero)) ) )
                
        ll /= observed.shape[0]
        
        
        return(-ll)


class negativeBernoulliGammaLogLikelihoodV2(nn.Module):

    def __init__(self, wet_day_threshold=1.0, epsilon_if_zero=1e-6, localize=False):
        super().__init__()
        self.wet_day_threshold = wet_day_threshold
        self.epsilon = epsilon_if_zero
        self.localize = False # this is always false

    def forward(self, observed, predicted):
        # transforms the observed data to avoid the location
        # parameters
        observed = torch.where(
            observed > self.wet_day_threshold, 
            torch.sub(observed, self.wet_day_threshold), 0.0
        )

        p = predicted[:, [0], ...]
        log_shape = predicted[:, [1], ...]
        log_scale = predicted[:, [2], ...]

        shape = torch.exp(log_shape)
        scale = torch.exp(log_scale)

        bool_rain = (observed > 0).float()

        no_rain = (1 - bool_rain) * torch.log(1 - p + self.epsilon)
        rain    = bool_rain * (
            torch.log(p + self.epsilon)
            + (shape - 1) * torch.log(observed + self.epsilon)
            - shape * torch.log(scale + self.epsilon)
            - torch.lgamma(shape + self.epsilon)
            - observed / (scale + self.epsilon)
        )

        loss = -torch.mean(no_rain + rain)
        return loss

