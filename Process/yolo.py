# 파이썬, 파이토치, Cuda, cuDNN 버전
import sys
print("Python version: {}".format(sys.version))

import torch
print("Torch version: {}".format(torch.__version__))
# !pip3 install torch torchvision 설치 안 된 경우
print("Cuda version: {}".format(torch.version.cuda))
print("cuDNN version: {}".format(torch.backends.cudnn.version()))
# cuda 사용 가능 여부 확인
torch.cuda.is_available()