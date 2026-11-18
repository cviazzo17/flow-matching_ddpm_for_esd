# Flow Matching y DDPM para reducción de escala estadística: desarrollo y evaluación de modelos

Este repositorio contiene los códigos utilizados para desarrollar y evaluar los modelos de Flow Matching y DDPM
en la tarea de reducción de escala estadística.

La estructura del repositorio es la siguiente:

- `code_models/*`: Contiene todos los scripts utilizados para el entrenamiento y la simulación de la UNet (proporciona
  conditioning para Flow Matching y DDPM), Flow Matching y DDPM. [code_models/README.md](https://github.com/cviazzo17/flow-matching_ddpm_for_esd/blob/main/code_models/README.md)
  proporciona información sobre el contenido del este subdirectorio.

- `code_evaluation/*`: Contiene todos los scripts utilizados para la evaluación de los modelos. [code_evaluation/README.md](https://github.com/cviazzo17/flow-matching_ddpm_for_esd/blob/main/code_evaluation/README.md)
  proporciona información sobre el contenido del este subdirectorio.

Los predictores y el target fueron extraidos del reanálisis ERA5 ([Hersbach et al., 2020](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803)) y pueden ser descargados en Copernicus Data Store (CDS; https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview 
y https://cds.climate.copernicus.eu/datasets/derived-era5-pressure-levels-daily-statistics?tab=overview).

Este repositorio es una extensión del código desarrollados por [Legasa et el. (2026)](https://arxiv.org/abs/2606.14570) (https://github.com/MNLR/Diffusion/tree/main).

## Referencias

- Hersbach H, Bell B, Berrisford P, et al. The ERA5 global reanalysis. Q J R Meteorol Soc. 2020; 146: 1999–2049. https://doi.org/10.1002/qj.3803
- Legasa M N, Doury A, Gellens A, Lguensat R, Naldesi C, Thao M, Vrac M. Preprint. 2026. https://arxiv.org/abs/2606.14570

