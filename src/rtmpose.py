"""ONNX inference for the RTMPose Halpe26 model on pre-cropped players."""

from pathlib import Path

import cv2
import numpy as np

RTMPOSE_BACKEND_TYPE = "rtmpose.halpe26.onnxruntime"
DEFAULT_RTMPOSE_MODEL_NAME = "rtmpose-l-halpe26-384x288-fp16.onnx"
DEFAULT_RTMPOSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / DEFAULT_RTMPOSE_MODEL_NAME
RTMPOSE_CROP_PADDING = 0.15  # Fraction of the tracked box added to each edge.

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def open_session(model, device):
    """Open the requested execution provider and fail if CUDA cannot initialize."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    device_name = "CPU"
    if device == "cuda":
        import torch  # Load CUDA libraries before creating the ONNX session.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        providers.insert(0, "CUDAExecutionProvider")
        device_name = torch.cuda.get_device_name()
    elif device != "cpu":
        raise ValueError(f"Unsupported RTMPose device: {device}")
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError(f"ONNX Runtime could not initialize {providers[0]}: {session.get_providers()}")
    return session, device_name


def validate_fp16_halpe26_model(session):
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(float16)" or model_input.shape[1:] != [3, 384, 288]:
        raise ValueError("expected an FP16 RTMPose-L Halpe26 ONNX input at 384x288")
    if isinstance(model_input.shape[0], int) and model_input.shape[0] != 1:
        raise ValueError("RTMPose must support dynamic batching or batch size 1")
    return model_input.shape[0]


def prepare(crop, input_width, input_height):
    height, width = crop.shape[:2]
    scale = min(input_width / width, input_height / height)
    transform = np.array([[scale, 0, (input_width - width * scale) / 2],
                          [0, scale, (input_height - height * scale) / 2]])
    rgb = cv2.warpAffine(crop, transform, (input_width, input_height))[:, :, ::-1]
    tensor = ((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return tensor, cv2.invertAffineTransform(transform)


def predict(session, crops):
    model_input = session.get_inputs()[0]
    _, _, height, width = model_input.shape
    input_types = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
    if model_input.type not in input_types:
        raise ValueError(f"Unsupported RTMPose input type: {model_input.type}")
    if model_input.shape[0] == 1 and len(crops) > 1:
        return [predict(session, [crop])[0] for crop in crops]
    tensors, transforms = zip(*(prepare(crop, width, height) for crop in crops))
    batch = np.stack(tensors).astype(input_types[model_input.type], copy=False)
    x, y = session.run(["simcc_x", "simcc_y"], {model_input.name: batch})
    if x.shape != (len(crops), 26, 2 * width) or y.shape != (len(crops), 26, 2 * height):
        raise ValueError(f"Expected Halpe26 SimCC outputs at {height}x{width}, got {x.shape} and {y.shape}")
    scores = np.minimum(x.max(axis=-1), y.max(axis=-1)).astype(np.float32)
    points = np.stack([x.argmax(axis=-1), y.argmax(axis=-1)], axis=-1) / 2
    return [(xy @ inverse[:, :2].T + inverse[:, 2], score)
            for xy, score, inverse in zip(points, scores, transforms)]
