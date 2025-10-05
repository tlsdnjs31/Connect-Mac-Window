import torch
import torch.nn as nn # 자동 손실, 정확도 계산해주는 라이브러리
import torch.optim as optim # 최적화 알고리즘 라이브러리
from torchvision import transforms, models
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "public", "resnet18_mnist.pth")

transform = transforms.Compose([
    transforms.Resize(224),                # ResNet18 입력 크기 224x224
    transforms.Grayscale(num_output_channels=3),  # 1채널 → 3채널
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

feature_maps = {}
fc = {}
# last_result = {}
last_result = None

def load_model():
    model = models.resnet18(weights=None)  # 학습용 ResNet18
    model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST 클래스 수: 10

    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()    # 모델을 evaluation 모드로 전환

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

# hook 함수 정의
def register_hooks(model, feature_maps, fc):
    def save_feature(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook

    def save_fc(name):
        def hook(module, input, output):
            fc[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(save_feature(name))

    model.fc.register_forward_hook(save_fc("fc"))

def process_image(pil_image, model):
    tensor = transform(pil_image).unsqueeze(0)
    _ = model(tensor)   # forward 실행 → hook으로 feature 저장

def normalization(arr):
    arr = arr.cpu().numpy()
    arr_min = arr.min()
    arr_max = arr.max()

    # 0~255 정규화
    normalized = ((arr - arr_min) / (arr_max - arr_min)) * 255
    normalized = normalized.astype(np.uint8) 
    return normalized


# def get_normalized_outputs(pil_image=None):
#     global last_result

#     if pil_image is not None:
#         model = load_model()
#         register_hooks(model, feature_maps, fc)
#         process_image(pil_image, model)

#         fmap_out = {}
#         for layer_name, fmap in feature_maps.items():
#             arr = normalization(fmap)   # numpy 배열 (보통 (C,H,W) 또는 (H,W))
#             arr = arr.squeeze()         # 불필요한 차원 제거
#             if arr.ndim == 3:           # (C, H, W)
#                 arr = arr[0]            # 첫 번째 채널만
#             if arr.ndim == 2:           # (H, W)
#                 arr = arr[0:1, :]       # 첫 번째 행만
#             fmap_out[layer_name] = arr.tolist()

#         # fc는 Dense라 그냥 전체 출력
#         fc_out = {
#             layer_name: normalization(fmap).tolist()
#             for layer_name, fmap in fc.items()
#         }

#         last_result = {"layers": {**fmap_out, **fc_out}}

#     return last_result




def get_normalized_outputs(pil_image=None):
    global last_result

    if pil_image is not None:
        model = load_model()
        register_hooks(model, feature_maps, fc)
        process_image(pil_image, model)

        fmap_out = {
            layer_name: normalization(fmap).tolist()
            for layer_name, fmap in feature_maps.items()
        }
        fc_out = {
            layer_name: normalization(fmap).tolist()
            for layer_name, fmap in fc.items()
        }
        last_result = {"layers": {**fmap_out, **fc_out}}

    return last_result


# def get_normalized_outputs(pil_image):
#     if pil_image is not None:
#         model = load_model()
#         register_hooks(model, feature_maps, fc)
#         process_image(pil_image, model)

#     fmap_out = {
#         layer_name: normalization(fmap).tolist()
#         for layer_name, fmap in feature_maps.items()
#     }
#     fc_out = {
#         layer_name: normalization(fmap).tolist()
#         for layer_name, fmap in fc.items()
#     }
#     last_result = {"layers": {**fmap_out, **fc_out}}
    
#     return last_result
#     # return fmap_out, fc_out
