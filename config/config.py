'''
Description: config for build model
Author: YuanJiang
Date: 2024-07-07 11:20:56
'''
from dataclasses import dataclass

####do not modify this class####
@dataclass
class BaseConfig:
    in_channel: int = 3
    image_height: int = 64

# TODO: modify the config for model build
@dataclass
class ModelConfig(BaseConfig):
    num_classes: int = 8
    act: str = 'gelu'
    kernel_size: int = 11
    dilation: int = 1
    conv: str = 'edge'
    norm: str = None
    bias: bool = True
    stochastic: bool = False
    epsilon: float = 0.0
    r: int = 1
    drop_path: float = 0.0
    channel_list: tuple = (64, 128, 256)

@dataclass
class TrainConfig(BaseConfig):
    #不要包含中文目录
    train_path: str = r'C:\Users\yuanjiang\Desktop\font\dataset_240610\train'
    test_path: str = r'C:\Users\yuanjiang\Desktop\font\dataset_240610\test'
    weights: str = ''
    freeze_layers: bool = False
    epochs: int = 100
    batch_size: int = 16
    lr: float = 0.01
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    device='cuda:0'


if __name__ == '__main__':
    config = ModelConfig()
    print(config)