# Fine-Tuning Whisper Model for Hugging Face and Custom Dataset

This repository contains the code for fine-tuning the Whisper ASR (Automatic Speech Recognition) model for both Hugging Face's datasets and custom datasets. Whisper is a deep learning model for ASR developed by Facebook AI. This README provides instructions for setting up and using this repository.

## Repository Structure

The repository is organized as follows:

- **custom_data_sample**: This directory contains a sample dataset for custom data. You can replace this with your own custom data or create a directory structure as shown below when fine-tuning the model

- **finetune**: The core directory for the fine-tuning process contains the following subdirectories and Python scripts:

  - **constant**: This directory contains hyperparameter constants used during fine-tuning.

  - **engine**: The core directory for fine-tuning. It contains the following Python scripts:

    - `data_setup.py`: Preprocesses data and prepares custom data when custom data is provided. It handles data loading, feature extraction, and data preparation.

    - `eval_engine.py`: Provides functionalities for model evaluation, such as calculating performance metrics (Word Error Rate).

    - `prep_custom_data.py`: Contains functionalities for preparing custom data. The structure in which custom data should be formatted for fine-tuning is given below.

    - `trainer.py`: Contains the code for running the training process based on the model and data provided.

  - **entity**: Contains scripts for validating input data and ensuring proper data formatting.

  - **utils**: Provides utility functions and helper scripts.

- **train.py**: This script runs the entire training pipeline for fine-tuning the Whisper model.

- **evaluate_hf_data.py**: This script is used to evaluate the model's performance on Hugging Face's datasets.

- **evaluate_custom_data.py**: This script is used to evaluate the model's performance on custom datasets. The structure in which custom data should be structured is given below.

- **transcribe.py**: Allows you to generate transcripts using the fine-tuned Whisper ASR model.

## Getting Started

To use this repository for fine-tuning the Whisper model, follow these steps:

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/your-username/whisper-finetuning.git
   ```

2. Install the necessary dependencies by running:

   ```
   pip install -r requirements.txt
   ```

3. Prepare your custom data. Make sure it follows the structure described in `prep_custom_data.py`.

4. Configure hyperparameters and other settings in the `constant` directory.

5. Run the training pipeline using the `train.py` script. You can specify the dataset you want to use for fine-tuning (Hugging Face or custom) by modifying the script accordingly.

6. Use the evaluation scripts (`evaluate_hf_data.py` and `evaluate_custom_data.py`) to assess the model's performance on the respective datasets.

7. Utilize the `transcribe.py` script to generate transcripts using the fine-tuned Whisper model.

## Custom Data Format

When preparing custom data, make sure it follows the structure defined below. The script will handle data preprocessing and formatting to make it compatible with the Whisper model.

```
data_dir/
├── audio/
│   ├── collection_1/
│   │   ├── (audio files related to collection_1)
│   ├── collection_2/
│   │   ├── (audio files related to collection_2)
│   ├── collection_3/
│   │   ├── (audio files related to collection_3)
├── transcripts/
│   ├── collection_1/
│   │   ├── transcript.csv (transcriptions for collection_1 audio)
│   ├── collection_2/
│   │   ├── transcript.csv (transcriptions for collection_2 audio)
│   ├── collection_3/
│   │   ├── transcript.csv (transcriptions for collection_3 audio)
```

#### Transcript Format
Each sub-folder within the "transcripts" directory contains a `transcript.csv` file. This CSV file has two columns:

1.  `id`: This column contains the unique identifier for each audio file, which corresponds to the audio file name.
2.  `transcription`: This column contains the respective transcriptions for the audio files identified by their `id`.


## License

This repository is provided under the MIT License. Please refer to the `LICENSE` file for more details.

For more information on the Whisper ASR model, you can visit the [Facebook AI Whisper GitHub repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/whisper) for additional resources and documentation.

If you encounter any issues or have questions, feel free to create an issue in this repository for support.

Happy fine-tuning with Whisper! 🎙️🤖