from ultralytics import YOLO

def main():
    model = YOLO('runs/style/weights/best.pt')

    model.predict(
        source='dataset/images/test',  # test 이미지 폴더
        conf=0.4,
        save=True,
        project='runs',
        name='style_test_pred'
    )

if __name__ == '__main__':
    main()
