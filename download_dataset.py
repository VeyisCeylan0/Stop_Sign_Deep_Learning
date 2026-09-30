
import argparse
import os

from roboflow import Roboflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="sign-detection-h24ey")
    p.add_argument("--project", default="stop-sign-h0vwm")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", default="datasets/stop_sign")
    args = p.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY ortam değişkenini ayarla.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    project.version(args.version).download("yolov8", location=args.out)
    print(f"Veri seti indirildi: {args.out}")


if __name__ == "__main__":
    main()
