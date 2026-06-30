"""Task taxonomy.

Two orthogonal questions about a task:
  * What MODALITY are its inputs (text / image / audio)? -> how we feed inputs.
  * What does it OUTPUT (free text vs. a class label)?   -> which metrics apply.
"""

from __future__ import annotations

# Input modality -> how we feed/resolve a task's inputs. Anything not listed
# here is treated as text.
IMAGE_INPUT_TASKS = {"image-classification", "image-to-text", "image-segmentation"}
AUDIO_INPUT_TASKS = {"automatic-speech-recognition", "audio-classification"}

# Output style -> how we read a prediction and which metrics apply.
# Generation tasks emit free text (compared with WER/BLEU/ROUGE); everything
# else is treated as classification (compared with accuracy/precision/recall/F1).
GENERATION_TASKS = {"image-to-text", "automatic-speech-recognition"}


def input_modality(task: str) -> str:
    """Return the input modality of a task: 'text', 'image', or 'audio'."""
    if task in IMAGE_INPUT_TASKS:
        return "image"
    if task in AUDIO_INPUT_TASKS:
        return "audio"
    return "text"


def is_generation_task(task: str) -> bool:
    """True if the task emits free text rather than a class label."""
    return task in GENERATION_TASKS
