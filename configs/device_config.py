

import torch


class DeviceConfig:
    use_gpu = True
    gpu_ids = [0]
    cuda_visible_devices = "0"
    dtype = torch.float32
    use_amp = True
    
    @classmethod
    def get_device(cls):
        if cls.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{cls.gpu_ids[0]}")
        else:
            return torch.device("cpu")

