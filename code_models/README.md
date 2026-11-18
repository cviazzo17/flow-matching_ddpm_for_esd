## Códigos de entrenamiento y simulación usados para desarrollar los modelos de deep learning

> Los códigos en el presente directorio son una extensión a Flow Matching del modelo DDPM desarrollado por [Legasa et al. (2026) (pre-print)](https://arxiv.org/abs/2606.14570). Los códigos de DDPM y la UNet desarrollado por Legasa están disponibles en [https://github.com/MNLR/Diffusion](https://github.com/MNLR/Diffusion/tree/main). 

- UNet determinística/estocástica: `EXP_BernoulliGamma.py`, `EXP_BernoulliGamma_Simulate.py`, y `CFG_BernoulliGammaUNET.py`.

- DDPM: `EXP_BGDiffusion.py`, `EXP_BGDiffusion_Simulate.py`, y `CFG_Def.py`.

- Flow Matching: `EXP_BGFlowMatching.py`, `EXP_BGFlowMatching_Simulate.py`, y `CFG_FlowMatching.py`.

`CFGD_StandardTransforms4Prediction_incX1D.py` contiene la estadarización utilizada en la UNet para los predictores. 

`CFGD_sqrTransforms.py` contiene la tranformación utilizada en DDPM y Flow Matching para los predictores y predictando.
