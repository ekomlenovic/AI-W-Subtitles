# Video Subtitle Generator

This project provides a tool for automatically generating subtitles for videos, supporting both transcription and translation into multiple languages using **Whisper** and **Helsinki-NLP MarianMT** models.

## Features

- Extracts audio from video files.
- Detects the language of the audio and transcribes it into subtitles.
- Optionally translates the subtitles to another language (English <-> French).
- Generates SRT subtitle files.

## Requirements

- Python 3.11+
- `whisper` (OpenAI's Whisper model)
- `torch` (PyTorch)
- `moviepy` (For video and audio processing)
- `transformers` (For translation models)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ekomlenovic/AI-W-Subtitles.git
   cd AI-W-Subtitles
   ```

2. Install the required packages with pip or conda:

   ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env create -f environment.yml
    conda activate subtitles
    ```

## Usage

The script can be run from the command line using the following syntax:

```bash
python main.py --translate <fr/en> ./path/to/video.<mp4/mkv/avi>
```

### Arguments:

- `--model_name`: (Optional) The Whisper model to use. Default is turbo. Available options: `tiny`, `base`, `small`, `medium`, `large`, `turbo`.
[Choose the model.](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)
- `--translate`: (Optional) The language to translate the subtitles to. Options: `fr`, `en` (French, English).

### Output
The generated subtitle files will be saved in a folder named `subtitles`, containing:

- Original subtitles in the detected language (e.g., `video_fr.srt`).
- Translated subtitles in the specified language (e.g., `video_en.srt`), if translation is requested.


### Documentation website generated with MkDocs in site folder
Building the documentation website.
```bash
mkdocs build
```
The documentation website is generated with MkDocs and is available in the `site` folder. To view the documentation, open the `index.html` file in your browser.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are always welcome!