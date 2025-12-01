import os
import random
import shutil

# ==== 1. 원본 데이터 경로 (GitHub에서 가져온 폴더) ====
# 예: 프로젝트 루트에 DataPattern-HW 폴더가 있다고 가정
ROOT = os.path.join(os.path.dirname(__file__), 'data')

IMAGE_DIR = os.path.join(ROOT, 'image')        # 원본 이미지 폴더
LABEL_DIR = os.path.join(ROOT, 'annotation')   # 원본 라벨(.txt) 폴더

# 우리가 만들 YOLO 학습용 폴더
OUT_ROOT = os.path.join(os.path.dirname(__file__), 'dataset')


def split_dataset(image_dir, label_dir, out_root,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 이미지 확장자 (png, jpg 둘 다 처리)
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"총 이미지 개수: {len(images)}")
    random.seed(42)
    random.shuffle(images)

    train_end = int(train_ratio * len(images))
    val_end = int((train_ratio + val_ratio) * len(images))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images,
    }

    for split_name, image_files in splits.items():
        out_img_dir = os.path.join(out_root, 'images', split_name)
        out_lbl_dir = os.path.join(out_root, 'labels', split_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        print(f"[{split_name}] 이미지 {len(image_files)}장 복사 중...")

        copied = 0
        for img_name in image_files:
            base = os.path.splitext(img_name)[0]
            lbl_name = base + '.txt'

            src_img = os.path.join(image_dir, img_name)
            src_lbl = os.path.join(label_dir, lbl_name)

            if not os.path.exists(src_lbl):
                print(f"⚠ 라벨 없음, 스킵: {img_name}")
                continue

            dst_img = os.path.join(out_img_dir, img_name)
            dst_lbl = os.path.join(out_lbl_dir, lbl_name)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            copied += 1

        print(f"✅ [{split_name}] 복사 완료: {copied}장")

    print("\n🎉 데이터 분할 완료!")
    print(f"dataset 구조: {out_root}")


if __name__ == '__main__':
    split_dataset(IMAGE_DIR, LABEL_DIR, OUT_ROOT)
