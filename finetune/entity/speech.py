from pydantic import BaseModel
from pathlib import Path


class SpeechSegment(BaseModel):
    model: str
    language: str
    hf_data: str | None
    hf_data_config: str | None
    is_custom_data: bool
    custom_data_path: Path | None
    prepare_custom_data: bool
    save_preprocessed_data: bool
    output_dir: Path
    training_strategy: str
