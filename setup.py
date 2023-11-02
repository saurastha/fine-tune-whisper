import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = 'fine-tune-whisper'
SRC_REPO = 'finetune'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description='Package setup for Whisper app',
    long_description=long_description,
    packages=setuptools.find_packages()
)
