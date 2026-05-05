"""
Phase C — ONNX INT8 Export

Runs on workstation (amd64) inside the training Docker container or GitHub Actions.
Downloads .pt weights from S3, exports to ONNX, quantizes to INT8, uploads back to S3.

Usage:
    python export_onnx.py --dataset-version v1 --run-id <mlflow-run-id>

Or in GitHub Actions (triggered automatically by ci_deploy.yml).
"""

import argparse
import os
import tempfile

import boto3
import mlflow
import torch
import yaml

S3_BUCKET = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
AWS_REGION = "eu-central-1"
MLFLOW_TRACKING_URI = "https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow"


def load_params(params_path: str = "pipeline/phase_c/detr/params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def download_weights_from_s3(dataset_version: str, local_path: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"weights/detr/{dataset_version}/model.pt"
    print(f"Downloading s3://{S3_BUCKET}/{key} → {local_path}")
    s3.download_file(S3_BUCKET, key, local_path)
    print("Download complete.")


class _DetrExportWrapper(torch.nn.Module):
    """Wraps ConditionalDetr so torch.onnx.export sees a single-tensor input."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def export_to_onnx(weights_path: str, onnx_dir: str, num_classes: int):
    """Export fine-tuned Conditional DETR to ONNX via torch.onnx.export."""
    print(f"Exporting to ONNX → {onnx_dir}")
    os.makedirs(onnx_dir, exist_ok=True)

    from transformers import AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/conditional-detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = _DetrExportWrapper(model)
    dummy = torch.randn(1, 3, 800, 800)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=16,
        do_constant_folding=True,
    )
    print(f"ONNX export complete → {onnx_path}")


def quantize_int8(onnx_dir: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = os.path.join(onnx_dir, "model.onnx")
    output_path = os.path.join(onnx_dir, "model_int8.onnx")

    print(f"Quantizing INT8: {input_path} → {output_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    print("INT8 quantization complete.")

    size_fp32 = os.path.getsize(input_path) / 1e6
    size_int8 = os.path.getsize(output_path) / 1e6
    print(f"  FP32: {size_fp32:.1f} MB → INT8: {size_int8:.1f} MB ({size_int8/size_fp32*100:.0f}%)")
    return output_path


def validate_onnx(onnx_path: str):
    import numpy as np
    import onnxruntime as ort

    print(f"Validating {onnx_path}...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 800, 800).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy})
    print(f"  Validation OK — outputs: {[o.shape for o in outputs]}")


def upload_to_s3(local_path: str, dataset_version: str, filename: str) -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"weights/detr/{dataset_version}/{filename}"
    print(f"Uploading {local_path} → s3://{S3_BUCKET}/{key}")
    s3.upload_file(local_path, S3_BUCKET, key)
    s3_url = f"s3://{S3_BUCKET}/{key}"
    print(f"Uploaded: {s3_url}")
    return s3_url


def log_to_mlflow(run_id: str, int8_path: str, s3_url: str, dataset_version: str):
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USERNAME", ""))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ.get("DAGSHUB_TOKEN", ""))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(int8_path, artifact_path="onnx")
        mlflow.log_param("onnx_int8_s3_url", s3_url)
        mlflow.log_param("quantization", "dynamic_int8")
        size_mb = os.path.getsize(int8_path) / 1e6
        mlflow.log_metric("model_int8_size_mb", size_mb)
    print(f"Logged ONNX artifact to MLflow run {run_id}")


def main():
    parser = argparse.ArgumentParser(description="Phase C ONNX INT8 export")
    parser.add_argument("--dataset-version", required=True, help="e.g. v1")
    parser.add_argument("--run-id", required=True, help="MLflow run ID from Colab training")
    parser.add_argument("--params", default="pipeline/phase_c/detr/params.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Use local weights.pt")
    args = parser.parse_args()

    params = load_params(args.params)
    num_classes = params["dataset"]["num_classes"]

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "model.pt")
        onnx_dir = os.path.join(tmpdir, "onnx")

        if not args.skip_download:
            download_weights_from_s3(args.dataset_version, weights_path)
        else:
            weights_path = "model.pt"

        export_to_onnx(weights_path, onnx_dir, num_classes)
        int8_path = quantize_int8(onnx_dir)
        validate_onnx(int8_path)

        s3_url = upload_to_s3(int8_path, args.dataset_version, "model_int8.onnx")
        log_to_mlflow(args.run_id, int8_path, s3_url, args.dataset_version)

    print(f"\nPhase C export complete.")
    print(f"INT8 model at: {s3_url}")
    print(f"Phase E Greengrass recipe should reference: {s3_url}")


if __name__ == "__main__":
    main()
