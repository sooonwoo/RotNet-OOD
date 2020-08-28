# Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty

Simplified implementation of Self-Supervised Learning for OOD Detection(NeurIPS 2019)

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 

full code(by authors): https://github.com/hendrycks/ss-ood

- command 
  - RotNet-OOD
  
    python test.py --method=rot --ood_dataset=cifar100
  
  - baseline
  
    python test.py --method=msp --ood_dataset=svhn



