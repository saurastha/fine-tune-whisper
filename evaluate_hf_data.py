import os
import re
import glob
import argparse
import pandas as pd
import evaluate
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='openai whisper model (tiny, small, medium..)')

parser.add_argument('--language',
                    help='language that is to be fine-tuned')

parser.add_argument('--hf_data',
                    default=None,
                    help='dataset name in of huggingface dataset')

parser.add_argument('--hf_data_config',
                    default=None,
                    help='like ne_np for nepali dataset in google/fleurs')

parser.add_argument('--task',
                    default='transcribe',
                    help='task to evaluate the data on (like transcribe, translate)')

parser.add_argument('--is_custom_data',
                    default=False,
                    help='set to True if fine tune is to be done on custom data')

parser.add_argument('--custom_data_path',
                    default=None,
                    help='Path to custom data')


parser.add_argument('--output_dir',
                    help='directory where the evaluation result is saved')

arguments = parser.parse_args()

config = ConfigurationManager(train=False)

eval_config = config.get_evaluation_config()

logger.info('>>>>>>>> Getting Configurations <<<<<<<<')
model_id = eval_config.model_path
pipe = pipeline("automatic-speech-recognition", model=model_id, device_map='auto')
wer_metric = load("wer")
normalizer = BasicTextNormalizer()

if pipe.device.type == 'cpu':
    logger.info(f'Note: CPU is used for inference. {eval_config.task} may take longer time.')


def download_audio(video: YouTube, save_path: str):
    audio = video.streams.filter(only_audio=True).first()
    audio.download(filename=save_path)
    logger.info(f'Audio saved in {save_path}')


def get_caption(video: YouTube):
    caption = video.captions['en']
    caption_xml = caption.xml_captions
    caption_str = xml_parser(caption_xml)
    return caption_str


def xml_parser(xml_file: ET) -> str:
    # Parse the XML content
    root = ET.fromstring(xml_file)

    # Extract the values of the 'p' attribute
    text_values = [p.text for p in root.findall('./body/p')]
    text = ' '.join(text_values)
    text = text.replace('"', '')
    return text


def transcribe_speech(filepath):
    output = pipe(
        filepath,
        generate_kwargs={
            "task": eval_config.task,
            "language": eval_config.language,
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]


if not os.path.exists(eval_config.root_dir):
    raise Exception("Folder not found: The specified folder for evaluation does not exist. "
                    "Please check the path and ensure that the folder exists before trying to access it.")

if not glob.glob('evaluation/*.csv'):
    raise Exception('CSV File not found for evaluation.')
else:
    df = pd.read_csv(glob.glob('evaluation/*.csv')[0])

urls = df['youtube_link'].tolist()

audio_save_dir = str(eval_config.root_dir) + '/' + 'audios'
if not os.path.exists(audio_save_dir):
    create_directories([audio_save_dir])

logger.info('>>>>>>>> Getting Audios <<<<<<<<')
for url in urls:
    try:
        pattern = r'v=([A-Za-z0-9_-]+)'
        found = re.search(pattern, url)
        video_id = found.group(1)
    except AttributeError:
        video_id = url.split('/')[-1]

    audio_save_path = str(audio_save_dir) + '/' + video_id + '.mp3'
    video = YouTube(url)
    download_audio(video=video, save_path=audio_save_path)
    caption = get_caption(video=video)
    norm_caption = normalizer(caption)

    logger.info(f'>>>>>>>> Transcribing audio from url {url} <<<<<<<<')
    prediction = transcribe_speech(audio_save_path)
    norm_prediction = normalizer(prediction)

    wer = wer_metric.compute(references=[norm_caption], predictions=[norm_prediction])

    idx = df.index[df['youtube_link'].str.contains(video_id)].tolist()
    df.loc[idx, 'actual_transcription'] = caption
    df.loc[idx, 'predicted_transcription'] = prediction
    df.loc[idx, 'WER'] = wer

df.to_csv(f'{eval_config.root_dir}/result.csv', index=False)

logger.info(f'Evaluation complete. Results save in {eval_config.root_dir}/result.csv')
