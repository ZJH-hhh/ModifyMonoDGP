import torch
ckpt = torch.load('outputs/monodgp/checkpoint_best.pth', map_location='cpu', weights_only=False)
print(ckpt.keys())          # 先看看有哪些键

# 如果看到有 'cfg' / 'args' / 'hyperparams' 之类
cfg = ckpt.get('model_state', None)
print(cfg)
