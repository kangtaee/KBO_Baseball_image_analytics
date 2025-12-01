import json
import uuid
from collections import Counter
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------
# 기본 경로 / 디렉토리 설정
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_JSON = BASE_DIR / "saved_results.json"

UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"

for d in [STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# 모델 로드 (경로는 본인 best.pt 위치에 맞게)
# ---------------------------------------------------
MODEL_PATH = BASE_DIR / "runs" / "style" / "weights" / "best.pt"

CLASS_NAMES = [
    "judge",
    "batter",
    "catcher",
    "pitcher",
    "infielder",
    "outfielder",
    "runner",
]

print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("[INFO] Model loaded.")

# ---------------------------------------------------
# FastAPI 세팅
# ---------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------
# Pydantic 모델 (이미지 예측 결과 저장용)
# ---------------------------------------------------
class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ImageResult(BaseModel):
    id: str
    original_filename: str
    input_image_url: str
    detections: List[Detection]


# ---------------------------------------------------
# saved_results.json 유틸
# ---------------------------------------------------
def load_saved_results() -> List[dict]:
    if not RESULTS_JSON.exists():
        return []
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] failed to load saved_results.json: {e}")
        return []


def save_saved_results(results: List[dict]) -> None:
    try:
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] failed to save saved_results.json: {e}")


# ---------------------------------------------------
# 메인 페이지
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    saved_results = load_saved_results()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "saved_results": saved_results,
        },
    )


# ---------------------------------------------------
# 이미지 예측 API
# ---------------------------------------------------
@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    ext = Path(file.filename).suffix or ".jpg"
    save_name = f"{uid}{ext}"
    save_path = UPLOAD_DIR / save_name

    # 원본 이미지 저장
    with open(save_path, "wb") as f:
        f.write(await file.read())

    print(f"[INFO] /api/predict - image file: {file.filename} -> {save_path}")

    # YOLO 추론 (이미지)
    results = model.predict(
        source=str(save_path),
        save=True,
        project=str(RESULTS_DIR),
        name=uid,
    )

    # YOLO가 그린 결과 이미지 찾기
    result_dir = RESULTS_DIR / uid
    result_img_path = None
    if result_dir.exists():
        for p in result_dir.iterdir():
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                result_img_path = p
                break

    # 못 찾으면 원본 이미지 사용
    if result_img_path is None:
        result_img_path = save_path

    # bbox 파싱
    detections: List[Detection] = []
    if len(results) > 0:
        r = results[0]
        boxes = r.boxes
        if boxes is not None and boxes.cls is not None:
            cls_list = boxes.cls.tolist()
            conf_list = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            for cls_idx, conf, box in zip(cls_list, conf_list, xyxy):
                cls_id = int(cls_idx)
                if 0 <= cls_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                x1, y1, x2, y2 = [int(v) for v in box]
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=round(float(conf), 4),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

    input_image_url = f"/static/{result_img_path.relative_to(STATIC_DIR).as_posix()}"

    res_json = {
        "id": uid,
        "original_filename": file.filename,
        "input_image_url": input_image_url,
        "detections": [d.dict() for d in detections],
    }

    return JSONResponse(content=res_json)


# ---------------------------------------------------
# 이미지 예측 결과 저장 API
# ---------------------------------------------------
@app.post("/api/save")
async def save_result(result: ImageResult):
    saved = load_saved_results()
    saved.append(result.dict())
    save_saved_results(saved)
    return {"status": "ok", "count": len(saved)}


# ---------------------------------------------------
# 저장된 결과 목록 API
# ---------------------------------------------------
@app.get("/api/saved")
async def get_saved():
    results = load_saved_results()
    return {"results": results}


# ---------------------------------------------------
# 동영상 예측 API (moviepy 없이, YOLO가 만든 비디오 그대로 사용)
# ---------------------------------------------------
@app.post("/api/predict_video")
async def predict_video(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex
    print("=== /api/predict_video called ===")
    print(f" - upload filename: {file.filename}")

    # 1) 업로드 동영상 저장
    upload_ext = Path(file.filename).suffix or ".mp4"
    upload_name = f"{request_id}{upload_ext}"
    upload_path = UPLOAD_DIR / upload_name

    with open(upload_path, "wb") as f:
        f.write(await file.read())
    print(f" - saved to: {upload_path}")

    # 2) YOLO로 동영상 추론
    results = model.predict(
        source=str(upload_path),
        save=True,
        project=str(RESULTS_DIR),
        name=request_id,
    )

    result_dir = RESULTS_DIR / request_id
    print(f" - result_dir: {result_dir}")

    # 3) YOLO가 저장한 결과 비디오(avi, mp4 등)를 찾기
    raw_video_path = None
    if result_dir.exists():
        for p in result_dir.iterdir():
            if p.suffix.lower() in [".avi", ".mp4", ".mov", ".mkv"]:
                raw_video_path = p
                break

    # 4) 요약(summary) 계산: 전체 프레임에서 클래스별 카운트
    summary = []
    try:
        counter = Counter()
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.cls is None:
                continue
            for cls_idx in boxes.cls.tolist():
                cls_id = int(cls_idx)
                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                counter[cls_name] += 1

        summary = [
            {"class_name": k, "count": int(v)} for k, v in counter.most_common()
        ]
    except Exception as e:
        print(f"[WARN] failed to build summary: {e}")

    # 5) 최종 video_url 생성 (변환 없이, YOLO가 만든 파일 그대로)
    video_url = None
    if raw_video_path is not None:
        print(f" - raw result video path: {raw_video_path}")
        rel = raw_video_path.relative_to(STATIC_DIR).as_posix()
        video_url = f"/static/{rel}"
    else:
        print("[WARN] No result video file found in result_dir")

    print(f" - final video_url: {video_url}")
    print("=== /api/predict_video done ===")

    return JSONResponse(
        content={
            "id": request_id,
            "original_filename": file.filename,
            "video_url": video_url,
            "summary": summary,
        }
    )
