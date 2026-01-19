# silly-sprites 🎭

A Python tool to automatically generate expression sprites for **SillyTavern** characters using Google's **Gemini AI**.

It takes a single "anchor" image of a character and generates 29 different emotional expressions (Joy, Anger, Sadness, Blush, etc.), maintaining the character's likeness while changing their facial expression and pose. It automatically removes the background to create ready-to-use transparent WebP sprites.

## Features

- **One-Shot Generation**: Creates a full sprite set from just one image.
- **SillyTavern Ready**: Generates files named correctly for ST (e.g., `joy.webp`, `anger.webp`) and supports the standard GoEmotions list.
- **Background Removal**: Uses `rembg` to automatically create transparent sprites.
- **Smart Resume**: Skips files that have already been generated, saving time and API credits.
- **Dramatic Poses**: Optional toggle to allow the AI to change the character's pose to match the emotion (e.g., slumping for sadness, hands up for surprise).
- **Web Interface**: Easy-to-use Gradio GUI with a gallery to view results.
- **Safety**: Configured to minimize refusals (CIVIC_INTEGRITY etc. set to BLOCK_NONE).

## Installation

### Option A: Using uv (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd silly-sprites
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

### Option B: Using standard pip

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd silly-sprites
    ```

2.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Web Interface (Recommended)

Run the startup script:

```bash
./startup.sh
```

Or run manually:

```bash
uv run sprites.py gui
```

1.  **Enter API Key**: Paste your Google Gemini API Key (it will be saved to `.env` for future use).
2.  **Upload Anchor**: Upload a clear image of your character.
3.  **Name**: Enter the character's name (this creates a folder in `./output/`).
4.  **Dramatic Poses**: Check this if you want more dynamic body language (unchecked = keep original pose).
5.  **Generate**: Click "🚀 Generate Sprites".

### CLI

You can also run it from the command line:

```bash
uv run sprites.py generate ./path/to/character.png --name "Vesper" --api-key "YOUR_KEY"
```

## Requirements

- Python 3.11+
- Google GenAI API Key

## License

MIT
