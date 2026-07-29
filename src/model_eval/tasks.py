"""Task taxonomy: modality + kind, with built-in defaults overridable via config."""

from __future__ import annotations

from dataclasses import dataclass

# task name -> (input modality, output kind). Unknown tasks default to text/classification.
_DEFAULTS: dict[str, tuple[str, str]] = {
    "sentiment-analysis": ("text", "classification"),
    "text-classification": ("text", "classification"),
    "image-classification": ("image", "classification"),
    "image-segmentation": ("image", "classification"),
    "image-to-text": ("image", "generation"),
    "audio-classification": ("audio", "classification"),
    "automatic-speech-recognition": ("audio", "generation"),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    modality: str = "text"  # text | image | audio
    kind: str = "classification"  # classification | generation


def resolve_task(name: str, config: dict | None = None) -> TaskSpec:
    """Resolve a task's modality/kind from defaults, overridable via config `tasks:`."""
    modality, kind = _DEFAULTS.get(name, ("text", "classification"))
    override = (config or {}).get("tasks", {}).get(name, {})
    return TaskSpec(name, override.get("modality", modality), override.get("kind", kind))
