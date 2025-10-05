# from PIL import Image
# import io
# from services.inside_return_featuremap import process_image, get_normalized_outputs

# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.gzip import GZipMiddleware
# from typing import List

# app = FastAPI()

# app.add_middleware(GZipMiddleware, minimum_size=1000)

# # 행렬 구조 반환 API
# # 각 레이어에 대한 행렬 값 제공
# @app.get("/api/inside/")
# def get_inside_layers():
#     pil_image = Image.open("")
#     fmap_out, fc_out = get_normalized_outputs(pil_image)
#     return{
#         "layers": {**fmap_out, **fc_out}
#     }

# # 클라이언트로부터 이미지를 업로드 받는 API
# # 미완성
# @app.post("/api/inside/")
# async def upload_inside_image(num_image: UploadFile = File(...)): # I/O 작업은 비동기로 처리
#     contents = await num_image.read()  # 업로드된 파일 읽기
#     pil_img = Image.open(io.BytesIO(contents))  # 바이트 데이터를 이미지로 변환
#     fmap_out, fc_out = get_normalized_outputs(pil_img)
#     return{
#         "layers": {**fmap_out, **fc_out}
#     }

from PIL import Image
import io
from services.inside_return_featuremap import get_normalized_outputs

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 마지막 결과 조회 API
@app.get("/api/inside/")
def get_inside_layers():
    result = get_normalized_outputs()   # pil_image 없이 호출 → last_result 반환
    if not result:
        return {"error": "아직 업로드된 이미지가 없습니다."}
    return result

# 이미지 업로드 후 결과 계산 API
@app.post("/api/inside/")
async def upload_inside_image(num_image: UploadFile = File(...)):
    contents = await num_image.read()
    pil_img = Image.open(io.BytesIO(contents))
    result = get_normalized_outputs(pil_img)   # 새 이미지로 계산
    return result
