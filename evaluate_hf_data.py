import argparse
from datasets import load_dataset
from finetune.engine.eval_engine import evaluate_model


def main():
    """
        Perform evaluation of a speech recognition model using specified parameters.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='Specify the Whisper model (e.g., tiny, small, medium) for evaluation.')

    parser.add_argument('--language',
                        help='Specify the target language for evaluation.')

    parser.add_argument('--task',
                        default='transcribe',
                        help='Specify the task to evaluate the data on (e.g., transcribe, translate).')

    parser.add_argument('--hf_dataset_id',
                        default=None,
                        help='Specify the dataset name from Hugging Face datasets for evaluation.')

    parser.add_argument('--hf_dataset_config',
                        default=None,
                        help='Specify the configuration (e.g., ne_np) for specific datasets like Nepali '
                             'in Google/fleurs.')

    parser.add_argument('--split',
                        default='test',
                        help='Specify the data split to be evaluated (e.g., test).')

    args = parser.parse_args()

    data = load_dataset(args.hf_dataset_id, args.hf_dataset_config, split=args.split)

    evaluate_model(model=args.model, language=args.language, task=args.task, data=data)


if __name__ == '__main__':
    main()
