from ultralytics import YOLO

def main():
    # 학습된 최고 성능 weight 불러오기
    model = YOLO('runs/style/weights/best.pt')

    # val 이미지에 대해 예측
    model.predict(
        source='dataset/images/val',  # val 이미지 폴더
        conf=0.4,                     # 신뢰도 threshold
        save=True,                    # 결과 이미지 저장
        project='runs',
        name='style_val_pred'         # runs/style_val_pred/ 에 저장됨
    )

if __name__ == '__main__':
    main()
