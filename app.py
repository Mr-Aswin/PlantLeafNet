"""Local Gradio application for PlantLeafNet disease classification."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "models" / "best_mobilenet_finetuned.pth"
CLASS_MAPPING_PATH = PROJECT_DIR / "class_mapping.json"
EXPECTED_CLASS_COUNT = 38
CONFIDENCE_THRESHOLD = 0.60
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class Prediction:
    """Prediction results prepared for display."""

    predicted_class: str
    confidence: float
    candidates: dict[str, float]
    message: str


def load_class_names(mapping_path: Path = CLASS_MAPPING_PATH) -> list[str]:
    """Load class names ordered by the numeric indices used by the model."""
    try:
        mapping: Any = json.loads(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Class mapping file was not found: {mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Class mapping is not valid JSON: {mapping_path}") from exc

    if not isinstance(mapping, dict) or len(mapping) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Class mapping must be an object containing exactly "
            f"{EXPECTED_CLASS_COUNT} classes."
        )

    indices: list[int] = []
    for class_name, class_index in mapping.items():
        if (
            not isinstance(class_name, str)
            or not class_name.strip()
            or isinstance(class_index, bool)
            or not isinstance(class_index, int)
        ):
            raise ValueError("Class mapping must contain non-empty names and integer indices.")
        indices.append(class_index)

    if sorted(indices) != list(range(EXPECTED_CLASS_COUNT)):
        raise ValueError(
            f"Class mapping indices must be unique and cover 0 through "
            f"{EXPECTED_CLASS_COUNT - 1}."
        )

    names = [""] * EXPECTED_CLASS_COUNT
    for class_name, class_index in mapping.items():
        names[class_index] = class_name
    return names


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint containers."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("The checkpoint does not contain a model state dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in checkpoint.items()):
        raise ValueError("The checkpoint state dictionary contains unsupported entries.")

    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def load_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
    mapping_path: Path = CLASS_MAPPING_PATH,
) -> tuple[nn.Module, list[str], torch.device, Any]:
    """Build MobileNetV3-Large and load the fine-tuned state dictionary."""
    class_names = load_class_names(mapping_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        load_options: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_options["weights_only"] = True
        checkpoint = torch.load(checkpoint_path, **load_options)
        state_dict = _checkpoint_state_dict(checkpoint)

        model = mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            len(class_names),
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load MobileNetV3-Large checkpoint at {checkpoint_path}: {exc}"
        ) from exc

    # This preset provides the official resize, center-crop, RGB conversion,
    # tensor conversion, and ImageNet normalization without downloading weights.
    preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()
    return model, class_names, device, preprocess


def _display_class_name(class_name: str) -> str:
    return class_name.replace("___", " — ").replace("_", " ")


def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    preprocess: Any,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Prediction:
    """Validate an uploaded image and return its top three model predictions."""
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload a valid JPG, JPEG, or PNG image.")
    if image.format is not None and image.format.upper() not in ALLOWED_IMAGE_FORMATS:
        raise ValueError("Unsupported image format. Please use JPG, JPEG, or PNG.")
    if len(class_names) < 3:
        raise ValueError("At least three mapped classes are required for top-three predictions.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    try:
        input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    except Exception as exc:
        raise ValueError(f"Could not preprocess the uploaded image: {exc}") from exc

    with torch.inference_mode():
        logits = model(input_tensor)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(class_names):
            raise ValueError(
                f"Model returned an unexpected output shape: {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=3)

    predicted_index = int(top_indices[0].item())
    confidence = float(top_probabilities[0].item())
    predicted_class = class_names[predicted_index]
    candidates = {
        _display_class_name(class_names[int(index.item())]): float(probability.item())
        for index, probability in zip(top_indices, top_probabilities)
    }

    if confidence < confidence_threshold:
        message = (
            f"⚠️ **Low confidence:** the leading candidate is "
            f"**{_display_class_name(predicted_class)}** "
            f"({confidence:.1%}), below the {confidence_threshold:.0%} threshold. "
            "Treat the candidates as uncertain."
        )
    else:
        message = (
            f"### {_display_class_name(predicted_class)}\n"
            f"**Confidence: {confidence:.1%}**"
        )

    message += "\n\nModel confidence is not a guarantee of diagnostic correctness."
    return Prediction(predicted_class, confidence, candidates, message)


def create_demo() -> gr.Blocks:
    """Create the Gradio application after loading and validating model files."""
    model, class_names, device, preprocess = load_model()

    def run_prediction(image: Image.Image | None) -> tuple[str, dict[str, float]]:
        if image is None:
            raise gr.Error("Upload a JPG, JPEG, or PNG leaf image to get a prediction.")
        try:
            result = predict_image(image, model, class_names, device, preprocess)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return result.message, result.candidates

    with gr.Blocks(title="PlantLeafNet – AI-Powered Plant Disease Detection") as demo:
        gr.Markdown(
            """
            # PlantLeafNet – AI-Powered Plant Disease Detection

            Upload a clear leaf photo to view the model's three leading class candidates.
            The model is a **MobileNetV3-Large** transfer-learning model fine-tuned for
            the **38 PlantVillage classes**.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Leaf image",
                    type="pil",
                    sources=["upload"],
                    image_mode="RGB",
                    height=360,
                )
                with gr.Row():
                    predict_button = gr.Button("Predict", variant="primary")
                    clear_button = gr.Button("Clear")
            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction")
                candidates_output = gr.Label(
                    label="Top three candidates",
                    num_top_classes=3,
                )

        gr.Markdown(
            f"""
            **Model information:** MobileNetV3-Large · PyTorch · {len(class_names)} classes ·
            PlantVillage · Transfer Learning + Fine-Tuning

            **Device:** {device}

            **Important:** The model was evaluated using the PlantVillage test split.
            Real-world field performance may differ due to lighting, background, camera
            quality, and environmental conditions. Predictions are decision-support
            information, not a guaranteed agricultural diagnosis. Model confidence does
            not mean diagnostic correctness is guaranteed.
            """
        )

        predict_button.click(
            fn=run_prediction,
            inputs=image_input,
            outputs=[prediction_output, candidates_output],
        )
        clear_button.click(
            fn=lambda: (None, "", {}),
            inputs=None,
            outputs=[image_input, prediction_output, candidates_output],
        )

    return demo


if __name__ == "__main__":
    create_demo().launch()
