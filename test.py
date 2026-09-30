"""Eğitilmiş modeli stop_sign_dataset (test seti) üzerinde çalıştırır.

Kullanım:
    python test.py
    python test.py --source datasets/stop_sign_dataset --conf 0.25
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 STOP sign testi")
    p.add_argument("--weights", default="runs/train/weights/best.pt", help="Eğitilmiş model")
    p.add_argument("--source", default="datasets/stop_sign_dataset",
                   help="Test görüntülerinin bulunduğu klasör")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Minimum güven eşiği; altındaki tespitler atılır")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU eşiği; üst üste binen kutuları eleme ölçütü")
    p.add_argument("--imgsz", type=int, default=640, help="Çıkarım görüntü boyutu")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    # Sonuç görselleri results/predictions altına kaydedilir (repoya eklenecek)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,
        project=str(Path("results").resolve()),
        name="predictions",
        exist_ok=True,
        stream=True,
    )

    total, detected, n_boxes = 0, 0, 0
    for r in results:
        total += 1
        n = len(r.boxes)
        n_boxes += n
        if n > 0:
            detected += 1
            confs = ", ".join(f"{float(c):.2f}" for c in r.boxes.conf)
            print(f"[+] {Path(r.path).name}: {n} tespit (conf: {confs})")
        else:
            print(f"[-] {Path(r.path).name}: tespit yok")

    print("\n--- Özet ---")
    print(f"Toplam görüntü        : {total}")
    print(f"Tespit yapılan görüntü: {detected}")
    print(f"Toplam kutu sayısı    : {n_boxes}")
    print(f"Conf={args.conf}, IoU={args.iou}")
    print("Görseller: results/predictions/")


if __name__ == "__main__":
    main()
