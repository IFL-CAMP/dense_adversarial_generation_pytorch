# Dense Adversarial Generation Pytorch
================

Main contributors: 

* Muhammad Ferjad Naeem
* Magdalini Paschali

![Alt text](sample.png?raw=true "DAG")

This repository implements the DAG attack proposed by Xie et al. **DAG** is an adversarial attack for semantic segmentation DNNs. The attack generates an adversarial image against a target which closely resembles the real image while fooling a state of the art segmentation DNN. The Jupyter Notebook contains an example of attacking a UNet network trained on the [Oasis](https://www.oasis-brains.org/) dataset.

### Requirements

* PyTorch > 0.4.0
* Numpy
* Matplotlib
* Random
* CUDA

### Publication
Paschali, M., Conjeti, S., Navarro, F., & Navab, N. (2018, September). Generalizability vs. robustness: investigating medical imaging networks using adversarial examples. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 493-501). Springer, Cham.

Xie, C., Wang, J., Zhang, Z., Zhou, Y., Xie, L., & Yuille, A. (2017). Adversarial examples for semantic segmentation and object detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1369-1378).