from pydantic import BaseModel
from pathlib import Path


class SpeechSegment(BaseModel):
    model: str
    language: str
    hf_data: str
    hf_data_config: str
    is_custom_data: bool = False
    custom_data_path: Path = None
    save_preprocessed_data: bool = True
    output_dir: Path = Path('data/')
    training_strategy: str
    resume_from_ckpt: bool = False
