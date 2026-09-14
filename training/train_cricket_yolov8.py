import os
import sys
import argparse
import yaml
import torch
from ultralytics import YOLO

def train_cricket_yolo(
    data_yaml_path: str,
    base_model: str = "yolov8m.pt",
    imgsz: int = 640,
    epochs: int = 50,
    batch: int = 16,
    project: str = "models/cricket/training_runs",
    name: str = "cricket_yolov8_run"
):
    """
    Fine-tunes YOLOv8 on cricket datasets (ball, stumps, bat, batsman)
    optimized for NVIDIA RTX 4060 GPU with CUDA FP16.
    """
    print("=" * 60)
    print(" SportsVision Cricket YOLOv8 Training Pipeline")
    print(f" CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Base Model: {base_model} | Img Size: {imgsz} | Epochs: {epochs} | Batch: {batch}")
    print("=" * 60)

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml_path}")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else 'cpu',
        half=torch.cuda.is_available(),  # FP16 mixed precision
        project=project,
        name=name,
        verbose=True,
        save=True,
        cache=True
    )

    print(f"\n[Training Complete] Best weights saved at: {results.save_dir}/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cricket YOLOv8 on RTX 4060 GPU")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset definition")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    train_cricket_yolo(
        data_yaml_path=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
