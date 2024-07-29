<p align="center" width="100%">
</p>

<div id="top" align="center">

Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/pdf/2405.18890"> 📑 Paper </a> |
<a href="https://github.com/MediaBrain-SJTU/FedLESAM"> 🐱 Github Repo </a> |
</h4>

<!-- **Authors:** -->

_**Ziqing Fan<sup>1,2 </sup>, Shengchao Hu<sup>1,2 </sup>, Jiangchao Yao<sup>1,2</sup>, Gang Niu<sup>3</sup>, Ya Zhang<sup>1,2</sup>, Masashi Sugiyama<sup>3,4</sup>, Yanfeng Wang<sup>1,2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory,
<sup>3</sup> RIKEN AIP,
<sup>4</sup> The University of Tokyo._

</div>


## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)


## Overview

In federated learning (FL), the multi-step update and data heterogeneity among clients often lead to a loss landscape with sharper minima, degenerating the performance of the resulted global model. Prevalent federated approaches incorporate sharpness-aware minimization (SAM) into local training to mitigate this problem. However, the local loss landscapes may not accurately reflect the flatness of global loss landscape in heterogeneous environments; as a result, minimizing local sharpness and calculating perturbations on client data might not align the efficacy of SAM in FL with centralized training. To overcome this challenge, we propose FedLESAM, a novel algorithm that locally estimates the direction of global perturbation on client side as the difference between global models received in the previous active and current rounds. Besides the improved quality, FedLESAM also speed up federated SAM-based
approaches since it only performs once backpropagation in each iteration. 

## Quick Start

Here we provide the implementation on Cifar-10 and Cifar100 datasets of following methods:

**FedAvg**: [Communication-Efficient Learning of Deep Networks
from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

**FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

**FedAdam**: [Adaptive Federated Optimization](https://openreview.net/pdf?id=LkFG3lB13U5)

**SCAFFOLD**: [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf)

**FedDyn**: [Federated Learning Based on
Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)

**FedCM**: [FedCM: Federated Learning with
Client-level Momentum](https://arxiv.org/pdf/2106.10874.pdf)

**FedSAM/MoFedSAM**: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)

**FedSkip(coming soon)**

**FedMR(coming soon)**

**FedGELA(coming soon)**

**FedLESAM, FedLESAM-S, FedLESAM-D**: [Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization](https://arxiv.org/pdf/2405.18890)

Here we provide a command to start the training of one algorithm:

**CUDA_VISIBLE_DEVICES=0 python train.py --non-iid --dataset CIFAR10 --model ResNet18 --split-rule Dirichlet --split-coef 0.6 --active-ratio 0.1 --total-client 100 --batchsize 50 --rho 0.5 --method FedLESAM-S --local-epochs 5 --comm-rounds 800**

For the best results, you might need to tune the parameter of rho.  
As for FedSMOO and FedGAMMA, the authors just make their codes open source. Please refer to the repo [FedSMOO](https://github.com/woodenchild95/FL-Simulator/tree/main), which might be more accurate for their algorithms. Notably, we try to implement our previous works FedSkip(ICDM22), FedMR(TMLR23) and FedGELA(NeurIPS23) in this repo. Feel free to use these methods for heterogeneous data in federated learning.

## Citation
If you find this work is relevant with your research or applications, please feel free to cite our work!
```
@inproceedings{FedLESAM,
  title={Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization},
  author={Fan, Ziqing and Hu, Shengchao and Yao, Jiangchao and Niu, Gang and Zhang, Ya and Sugiyama, Masashi and Wang, Yanfeng},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```

## Acknowledgments

This repo benefits from [FedSMOO](https://github.com/woodenchild95/FL-Simulator/tree/main). Thanks for their wonderful works!
