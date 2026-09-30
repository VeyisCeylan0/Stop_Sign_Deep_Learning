"""STOP tabelası tespiti için YOLOv8 eğitim scripti.

Kullanım:
    python train.py
    python train.py --data datasets/stop_sign/data.yaml --epochs 50 --batch 16
"""
import argparse

from ultralytics import YOLO
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 STOP sign eğitimi")
    p.add_argument("--data", default="datasets/stop_sign/data.yaml",
                   help="Roboflow'dan indirilen data.yaml yolu")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Pre-trained başlangıç ağırlığı (n/s/m/l/x)")
    p.add_argument("--epochs", type=int, default=50, help="Tüm veri setinin kaç kez görüleceği")
    p.add_argument("--batch", type=int, default=16, help="Bir adımda işlenen görüntü sayısı")
    p.add_argument("--imgsz", type=int, default=640, help="Giriş görüntü boyutu (px)")
    p.add_argument("--optimizer", default="AdamW", help="SGD, Adam, AdamW, auto ...")
    p.add_argument("--lr0", type=float, default=0.001, help="Başlangıç öğrenme oranı")
    p.add_argument("--patience", type=int, default=15,
                   help="Doğrulama metriği bu kadar epoch iyileşmezse eğitimi durdur")
    p.add_argument("--device", default=None, help="'0' (GPU), 'cpu' ... boşsa otomatik")
    return p.parse_args()


def main():
    args = parse_args()

    # COCO üzerinde önceden eğitilmiş (pre-trained) ağırlıkla başla -> transfer learning
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        project=str(Path("runs").resolve()),
        name="train",
        exist_ok=True,
        plots=True,  # results.png, confusion_matrix.png, F1/PR eğrileri
    )

    # Doğrulama seti üzerinde son metrikler
    metrics = model.val(project=str(Path("runs").resolve()), name="val", exist_ok=True)
    print("\n--- Doğrulama Sonuçları ---")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print("\nEn iyi ağırlık: runs/train/weights/best.pt")


if __name__ == "__main__":
    main()
