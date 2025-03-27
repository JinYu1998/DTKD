# Dynamic Temperature Knowledge Distillation

The paper link is <https://arxiv.org/abs/2404.12711>

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>).

# Errata and Clarification for DTKD Research

If you are reading DTKD, please take note: There are significant mathematical derivation errors in this paper. Specifically, the errors are in Formulas 4 and 5. The derivation contains a serious mistake. The correct $\delta$ should be $max_i {(u_i - v_i) / (u_i + v_i)} * t$. This means that the subsequent $\tau + \delta$ and $\tau - \delta$ cannot be derived as originally proposed. (Interestingly, this error was not detected by reviewers in two conference review processes. Thanks to Professor Xu for pointing out this error.)

However, if you directly use the previously derived results, it is still possible to achieve effective results in knowledge distillation for classification models. All DTKD data in the paper were personally run by me, and I can confirm the data is genuinely reliable.

Furthermore, if you discover that DTKD is effective in Large Language Model (LLM) distillation, I welcome you to email me. 

Note: My email address is weiyukang1998@163.com. The email address in the paper was previously written incorrectly.

## Framework & Performance

### Different teachers distilled into `ResNet8`
<div style="text-align:center"><img src=".github/resnet8.jpg" width="75%" ></div>

### Differernt teacher distilled into `MobileNetV2`
<div style="text-align:center"><img src=".github/mobilenetv2.jpg" width="75%" ></div>

## TODO
- [ ] To update the code that records the temperature in training.
- [ ] To update the analysis code.
- [ ] Release other network models. (Such as ResNetXXX)

## Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

For more details please refer to <https://github.com/megvii-research/mdistiller>

## CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  python3 tools/train_ours.py --cfg configs/cifar100/dtkd/res32x4_res8x4.yaml 
  ```

# Acknowledgement
- Sincere gratitude to the contributors of mdistiller for your distinguished efforts.

# Contact

YuKang Wei: weiyukang1998@163.com
