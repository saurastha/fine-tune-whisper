from pydantic import BaseModel
from pathlib import Path


class SpeechSegment(BaseModel):
    model: str = 'openai/whisper-small'
    language: str = 'english'
    hf_data: str = None
    hf_data_config: str = None
    is_custom_data: bool = False
    custom_data_path: Path = None
    save_preprocessed_data: bool = True
    output_dir: Path = Path('data/')
    training_strategy: str
