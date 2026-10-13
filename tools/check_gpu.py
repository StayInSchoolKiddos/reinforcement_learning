import subprocess
import sys
print('Running nvidia-smi...')
try:
    out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, text=True)
    print(out)
except Exception as e:
    print('nvidia-smi error:', e)

print('\nChecking PyTorch CUDA availability...')
try:
    import torch
    print('torch version:', torch.__version__)
    try:
        print('cuda_available:', torch.cuda.is_available())
        print('device_count:', torch.cuda.device_count())
        if torch.cuda.is_available():
            print('current_device:', torch.cuda.current_device())
            print('device_name:', torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as e:
        print('Error querying CUDA from torch:', e)
except Exception as e:
    print('torch import error:', e)
    print('sys.executable:', sys.executable)