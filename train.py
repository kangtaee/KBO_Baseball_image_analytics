from ultralytics import YOLO

def main():
    # 1) 사전 학습된 YOLOv8m 불러오기
    model = YOLO('yolov8m.pt')   # 처음 한 번은 자동으로 다운로드됨

    # 2) 학습 시작
    model.train(
        data='data.yaml',  # 우리가 방금 만든 yaml
        epochs=50,               # Velog도 50 에폭 기준
        imgsz=640,
        batch=16,
        project='runs',
        name='style',      # 결과가 runs/style/ 에 저장됨
        seed=123
    )

if __name__ == '__main__':
    main()
