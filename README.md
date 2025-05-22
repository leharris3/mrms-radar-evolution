# Diffusion-Powered MRMS Radar Evolution for Tornadic Storms

The Multi-Radar Multi-Sensor (MRMS) dataset provides a plethora of integrated radar and probabilistic products for forecasters and stakeholders. A natural idea is to use deep learning techniques to predict future radar states from previous ones, and to include some uncertainty quantification method as well. Diffusion-based weather-prediction models (e.g., GenCast) have proven their metal at medium-range scales. Now, there is ample opprotunity to evaluate similar techniques at fine-temporal resolutions to support nowcasting of dynamic severe weather events.

## Dataset

We use a subset of MRMS; inspired by the choices using in Deep Learning for *Probabilistic Nowcasting of Radar Imagery in Tornadic Storms* (Erikson et al.)

- Note: we use the pythonic convention for array indexing (e.g., [0:2] corresponds to elements {0, 1}).

###  Inputs

| Product                             |  Timesteps  |
| :---------------------------------- | :---------: |
| `MergedReflectivityComposite-0.5km` | `t=[-15:0]` |

###  Labels
| Product                             | Timesteps |
| :---------------------------------- | :-------: |
| `MergedReflectivityComposite-0.5km` | `t=[0:4]` |
