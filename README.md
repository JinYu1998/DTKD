## Code release for Dynamic Temperature Knowledge Distillation.

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>).

## TODO
- [ ] Release other network models. (Such as ResNetXXX)

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

For more details please refer to <https://github.com/megvii-research/mdistiller>

### CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  python3 tools/train_ours.py --cfg configs/cifar100/dtkd/res32x4_res8x4.yaml 
  ```

# Acknowledgement
- Sincere gratitude to the contributors of mdistiller for your distinguished efforts.

# Contact
[YuKang Wei]: weiyukang1998@163.com