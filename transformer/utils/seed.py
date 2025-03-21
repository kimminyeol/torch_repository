import torch
import numpy as np
import random

def set_seed(seed=42):
    """재현성을 확보하기 위한 랜덤 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU에서의 시드 고정
    np.random.seed(seed)
    random.seed(seed)