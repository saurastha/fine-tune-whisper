from pydantic import BaseModel
from pathlib import Path


class SpeechSegment(BaseModel):
    model: str
    language: str
    hf_dataset_id: str | None
    hf_dataset_config: str | None
    is_custom_audio_data: bool
    custom_audio_data_path: str | None
    prepare_custom_audio_data: bool
    save_preprocessed_data: bool
    output_dir: Path
    training_strategy: str
