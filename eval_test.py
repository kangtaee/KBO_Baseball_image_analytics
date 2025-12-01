from ultralytics import YOLO

def main():
    model = YOLO('runs/style/weights/best.pt')

    metrics = model.val(
        data='data.yaml',
        split='test'          # test 세트 기준 평가
    )

    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

if __name__ == '__main__':
    main()
