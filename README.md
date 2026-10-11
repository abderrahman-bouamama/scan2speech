# Scan2Speech

**An Automated Pipeline: Scanned PDFs → Clean Text → Studio Audiobooks → Full HD Videos**
**Scan2Speech: AI Scanned PDF to Studio Audiobooks & YouTube Videos Pipeline**
**Transform physical books & scanned PDFs into studio-grade voice-cloned audiobooks and 1080p documentary videos automatically.**

Created & Developed by **Abderrahmane Bouamama**  
License: **MIT**

---

## Overview

**Scan2Speech** is an end-to-end multimedia pipeline designed to transform noisy, scanned physical documents and PDF books into studio-grade audiobooks and Full HD documentary videos.

Traditional OCR tools produce fragmented text plagued by broken lines, running headers, footnotes, and hyphenated words, resulting in robotic, unlistenable speech. **Scan2Speech** solves this by pairing multimodal Vision AI (Google Gemini Flash / DeepSeek) with intelligent text post-processing, Coqui XTTS-v2 voice cloning, broadcast audio mastering, and automated video generation.

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   Scanned PDF    │ ────> │ High-Res Images  │ ────> │  Vision LLM OCR  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
                                                               │
┌──────────────────┐       ┌──────────────────┐                ▼
│  Multilingual    │ <──── │  Chapters Split  │ <─── │ Clean Transcript │
│ Polish/Tashkeel  │       └──────────────────┘       └──────────────────┘
│(OpenAI/DeepSeek) │
└──────────────────┘
        │
        ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  XTTS-v2 Voice   │ ────> │ EBU R128 Master  │ ────> │  Full HD 1080p   │
│     Cloning      │       │    Audiobook     │       │  Video & Slides  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

---

## Key Features

1. **Vision-Powered OCR (No PyTorch/CUDA required for OCR)**
   - Renders scanned PDF pages to high-resolution PNGs using `pypdfium2`.
   - Transcribes text using Google Gemini 1.5 / 2.0 / Flash or DeepSeek Vision.
   - Ultra-lightweight footprint (~30 MB core dependencies).
2. **Text Cleaning & Transcript Curation**
   - Strips standalone page numbers, dashes, borders, and running headers.
   - Removes bottom footnotes, divider lines, and in-text citation markers (`[1]`, `(1)`, `(١)`).
   - Rejoins hyphenated words and sentences broken across page breaks.
3. **Automated Chapter Division**
   - Splits continuous transcripts into structured chapters by headings or target word count.
   - Calculates word counts, character counts, and estimated audio narration runtimes.
4. **Multilingual Diacritization & Script Polishing**
   - Built-in prompts for the 6 most used languages: Arabic (Tashkeel/Harakat), English (speech normalization), French (accents & liaisons), Spanish (tildes & inverted punctuation), German (umlauts & pacing), and Italian (accents & rhythm).
   - Connects to any OpenAI-compatible API endpoint (DeepSeek, OpenAI, Groq, Ollama, OpenRouter, vLLM).
5. **Studio-Grade Neural Voice Cloning (XTTS-v2)**
   - Clones any speaker's voice from a short audio sample (`.wav` / `.mp3`).
   - Intelligent 120-character sentence chunking with smooth 5ms cosine fades (zero audio edge clicks).
   - Natural breath gaps (60ms) and paragraph pauses (280ms).
   - Audio mastering with broadcast filters and EBU R128 loudness normalization via FFmpeg.
   - Automated chapter concatenation and ID3 metadata tagging.
6. **YouTube-Ready Video Production**
   - **Static Cover Video**: Generates 1080p MP4 videos with artwork for individual chapters or master audiobooks.
   - **Documentary Slideshow**: Automatically synchronizes an audiobook with a folder of background images, distributing them evenly across the audio runtime.

---

## System Requirements & Configuration

### 1. Prerequisites

- **Python**: Version `3.10` or higher.
- **FFmpeg & FFprobe**: Required for audio mastering, concatenation, and video encoding.
  - **Windows**: Install via WinGet: `winget install Gyan.FFmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **NVIDIA GPU with CUDA** *(Optional, recommended for fast local voice cloning)*:
  - If no NVIDIA GPU is available, the XTTS-v2 synthesis falls back to CPU automatically.

### 2. Environment Variables & API Keys

Copy the sample environment file and configure your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# API Keys (at least one required for OCR)
GEMINI_API_KEY="your-google-gemini-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"

# Optional Pipeline Overrides
# SCAN2SPEECH_PROVIDER="gemini"       # "gemini" (default) or "deepseek"
# SCAN2SPEECH_MODEL="gemini-3.8-flash"
# SCAN2SPEECH_DPI_SCALE=2.0
# SCAN2SPEECH_RPM=10
```

| Variable | Description | When Needed |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | Scanned PDF OCR transcription |
| `DEEPSEEK_API_KEY` | DeepSeek API key | DeepSeek OCR & Arabic Tashkeel diacritization |
| `SCAN2SPEECH_PROVIDER` | AI vision provider (`gemini` or `deepseek`) | Optional override |
| `SCAN2SPEECH_DPI_SCALE` | DPI scale multiplier (default: `2.0`) | Image resolution adjustment |
| `SCAN2SPEECH_RPM` | Maximum requests per minute (default: `10`) | Rate-limit management |

---

## Installation

### Core OCR Pipeline (Lightweight)

```bash
git clone https://github.com/abderrahman-bouamama/scan2speech.git
cd scan2speech
pip install -e .
```

### Full Suite: Voice Cloning, Diacritization & Video Generation

To use Coqui XTTS-v2 voice cloning, Arabic diacritization, and video generation, install the optional audio dependencies:

```bash
pip install -r requirements-audio.txt
```

---

## Step-by-Step Usage Guide

### Step 1: Transcribe Scanned PDF to Clean Text

Run the main CLI against any scanned PDF document:

```bash
# Transcribe scanned document
scan2speech sample_book.pdf

# Specify language (e.g., Arabic, French, Spanish) and DPI scale
scan2speech sample_book.pdf --language ar --dpi-scale 2.5 --verbose
```

**Output:**
- `input/<pdf_name>/`: High-resolution rendered page images (`page_0001.png`, ...).
- `output/<pdf_name>/`: Cleaned per-page text files (`page_0001.txt`, ...) and `<pdf_name>_tts_ready.txt`.
- `.scan2speech_state.json`: Checkpoint manifest (resumes automatically if interrupted).

---

### Step 2: Build Curated Narration Transcript

Stitch the page transcripts while stripping running headers, footers, and footnote citations:

```bash
python build_audio_transcript.py \
  --input-dir output/sample_book \
  --title "My Audio Book" \
  --author "Abderrahmane Bouamama"
```

---

### Step 3: Split into Numbered Chapters

Divide the full text into organized, numbered chapter files for modular audiobook production:

```bash
# Split automatically by chapter headings or Markdown titles
python split_chapters.py \
  --input-file output/sample_book/sample_book_tts_ready.txt \
  --output-dir output/sample_book/chapters \
  --mode auto

# Or split by target word count (e.g., ~1,000 words per chapter ≈ 7.5 mins audio)
python split_chapters.py \
  --input-file output/sample_book/sample_book_tts_ready.txt \
  --output-dir output/sample_book/chapters \
  --mode words \
  --words-per-chapter 1000
```

---

### Step 4: Multilingual Text Polishing & Diacritization

Prepare your narration script with `diacritize_text.py`. It includes specialized system prompts for the **6 most used languages** and supports **any OpenAI-compatible API endpoint** (DeepSeek, OpenAI, Groq, Ollama, OpenRouter, vLLM):

| Language Code | Language | Purpose |
|---|---|---|
| `ar` | **Arabic (العربية)** | Full classical Tashkeel (Harakat) for broadcast recitation |
| `en` | **English** | Punctuation polishing, OCR artifact cleanup, spoken cadence |
| `fr` | **French (Français)** | Accent restoration (é, è, à, ç), liaisons, audio flow |
| `es` | **Spanish (Español)** | Diacritic accents (tildes), inverted punctuation (¿?, ¡!) |
| `de` | **German (Deutsch)** | Umlauts (ä, ö, ü, ß), compound word clarity, sentence pacing |
| `it` | **Italian (Italiano)** | Graphic accents (à, è, é), elisions, rhythmic audiobook pacing |

#### Running with Any Language

```bash
# Arabic: Full Tashkeel diacritization on all chapters
python diacritize_text.py --input-dir output/sample_book/chapters --language ar

# English: Polish narration script for natural TTS flow
python diacritize_text.py --input-dir output/sample_book/chapters --language en

# French: Restore accents and polish audiobook script
python diacritize_text.py --input-dir output/sample_book/chapters --language fr

# Spanish: Ensure accents and inverted punctuation
python diacritize_text.py --input-dir output/sample_book/chapters --language es

# German: Restore umlauts and cadence
python diacritize_text.py --input-dir output/sample_book/chapters --language de

# Italian: Polish pacing and accents
python diacritize_text.py --input-dir output/sample_book/chapters --language it
```

#### Customizing the API Endpoint (OpenAI-Compatible)

You are **not limited to DeepSeek**; you can use any provider or local model:

1. **Via `.env`**: Set `LLM_API_URL` (see `.env.example`).
2. **Via CLI Argument**: Pass `--api-url <URL>` and `--model <MODEL_NAME>`.
3. **Directly in Code**: Modify `DEFAULT_API_URL` or line in `diacritize_text.py`.

```bash
# Example with OpenAI GPT-4o-mini
python diacritize_text.py \
  --input-dir output/sample_book/chapters \
  --api-url "https://api.openai.com/v1/chat/completions" \
  --api-key "your-openai-key" \
  --model "gpt-4o-mini" \
  --language en

# Example with Groq (ultra-fast LLaMA 3.3)
python diacritize_text.py \
  --input-dir output/sample_book/chapters \
  --api-url "https://api.groq.com/openai/v1/chat/completions" \
  --api-key "your-groq-key" \
  --model "llama-3.3-70b-versatile" \
  --language fr

# Example with Local Ollama (no API key needed)
python diacritize_text.py \
  --input-dir output/sample_book/chapters \
  --api-url "http://localhost:11434/v1/chat/completions" \
  --api-key "ollama" \
  --model "llama3.2" \
  --language es
```

---

### Step 5: Audition Cloned Voice Sample

Test your reference voice sample with XTTS-v2 before launching full synthesis:

```bash
# Audition Arabic voice
python test_voice.py \
  --voice input/voice_reference.wav \
  --language ar

# Audition English voice
python test_voice.py \
  --voice input/voice_reference.wav \
  --language en \
  --text "Welcome to this audio audition of our neural voice model."
```

---

### Step 6: Generate Studio Audiobook (XTTS-v2)

Synthesize chapters using the cloned voice and master the audio with EBU R128 loudness normalization:

```bash
# Synthesize all chapters and merge into a master audiobook
python generate_audiobook.py \
  --input-dir output/sample_book/chapters \
  --voice input/voice_reference.wav \
  --title "My Audiobook" \
  --author "Abderrahmane Bouamama" \
  --language ar \
  --all

# Or synthesize a single chapter (e.g., Chapter 1)
python generate_audiobook.py \
  --input-dir output/sample_book/chapters \
  --voice input/voice_reference.wav \
  --chapter 1

# Merge already-generated chapter MP3s into the master MP3 without re-synthesizing
python generate_audiobook.py \
  --input-dir output/sample_book/chapters \
  --voice input/voice_reference.wav \
  --merge-only
```

---

### Step 7: Create Full HD Videos (Cover & Slideshow)

#### A. Static Cover Art Video

Combine an audio file with artwork to produce a 1080p MP4:

```bash
python generate_video.py \
  --audio output/audiobook/My_Audiobook_master.mp3 \
  --cover input/cover_art.jpg \
  --output output/videos/my_audiobook.mp4
```

#### B. Documentary Slideshow Video

Synchronize your audiobook with multiple background images distributed evenly across the narration runtime:

```bash
python generate_slideshow_video.py \
  --audio output/audiobook/My_Audiobook_master.mp3 \
  --images-dir input/documentary_photos/ \
  --output output/videos/my_documentary.mp4
```

---

## Repository Structure

```
scan2speech/
├── src/
│   └── scan2speech/
│       ├── __init__.py            # Package entry & custom exceptions
│       ├── __main__.py            # python -m scan2speech support
│       ├── cli.py                 # Command line interface
│       ├── config.py              # Configuration & environment loading
│       ├── pdf_splitter.py        # PDF → PNG image extraction (pypdfium2)
│       ├── vision_client.py       # Vision LLM client (Gemini / DeepSeek)
│       ├── text_cleaner.py        # OCR text cleaning & stitching
│       └── state.py               # State checkpointing & resume logic
├── build_audio_transcript.py      # Transcript curation (header/footer removal)
├── split_chapters.py              # Chapter division & runtime estimator
├── diacritize_text.py             # Arabic Tashkeel diacritization (DeepSeek)
├── generate_audiobook.py          # Studio XTTS-v2 voice cloning & EBU R128 mastering
├── test_voice.py                  # Voice audition & sample testing
├── generate_video.py              # Cover art MP4 video generator
├── generate_slideshow_video.py    # Multi-image documentary video generator
├── tests/                         # Pytest test suite (38 unit tests)
├── .env.example                   # Template configuration file
├── requirements.txt               # Core OCR dependencies
├── requirements-audio.txt         # Audio/video extensions dependencies
├── pyproject.toml                 # Packaging & project metadata
└── README.md
```

---

## Running Tests

Verify the core OCR pipeline:

```bash
pytest tests/ -v
```

---

## Author & Credits

- **Creator & Maintainer**: **Abderrahmane Bouamama**
- **GitHub**: [@abderrahman-bouamama](https://github.com/abderrahman-bouamama)
- **Email**: bouamama.abderrahmane95@gmail.com

---

## License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it for personal and commercial projects.
