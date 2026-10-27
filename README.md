# Rustset

Turn public Rust GitHub repositories into high-quality question-to-code instruction datasets for LLM fine-tuning.

Rustset clones specified Rust repositories, extracts eligible code, prompts any OpenAI-compatible model to synthesize developer instructions for each snippet, and packages the results into a Parquet file with an automatic Hugging Face dataset card.

---

## ✨ Features

- **Automated Filtering**: Clones and strips non-Rust source, generated code, and test directories (optional inclusion).
- **Flexible LLM Backends**: Works with OpenAI (`gpt-4o-mini`, etc.) or any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, LocalAI).
- **Configurable Chunking**: Control chunk sizes, minimum lines, and random sampling caps to control API costs.
- **Hugging Face Hub Integration**: Uploads Parquet dataset shards and custom dataset cards directly to Hugging Face.
- **Interactive Web Dashboard**: Built-in dark UI with real-time progress, log streaming, and Cloudflare Quick Tunnel support.
- **Headless CLI**: Scriptable for batch workflows and headless environments.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ha466/rust.git
cd rust
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example configuration:

```bash
cp .env.example .env
```

Set your API credentials in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
HF_TOKEN=hf_...
# Optional: custom endpoint (e.g. Ollama, OpenRouter, vLLM)
# OPENAI_BASE_URL=http://localhost:11434/v1
```

---

## 🖥️ Web Dashboard

Start the built-in browser console locally:

```bash
python ui.py --host 127.0.0.1 --port 7860
```

Open `http://localhost:7860` to configure source repositories, customize the dataset README template, and start pipeline runs interactively.

---

## ☁️ Run in Google Colab or Kaggle

You can run Rustset on cloud notebooks with a free, authenticated Cloudflare Quick Tunnel:

### Google Colab

```python
!git clone https://github.com/ha466/rust.git rustset
%cd rustset
!CLOUDFLARE_ALLOWED_EMAIL="you@example.com" bash run_colab_kaggle.sh
```

### Kaggle

```python
!git clone https://github.com/ha466/rust.git /kaggle/working/rustset
%cd /kaggle/working/rustset
!CLOUDFLARE_ALLOWED_EMAIL="you@example.com" bash run_colab_kaggle.sh
```

> **Note**: Replace `you@example.com` with your email. Cloudflare sends a one-time verification code when accessing the `trycloudflare.com` tunnel URL.

---

## ⚡ CLI Usage

### Generate & Upload to Hugging Face

```bash
python main.py BurntSushi/ripgrep \
  --api-key YOUR_API_KEY \
  --hf-repo YOUR_USER/rust-dataset \
  --hf-token YOUR_HF_TOKEN
```

### Local Only (No Upload)

Write Parquet and metadata directly to `out/` without uploading:

```bash
python main.py tokio-rs/tokio --no-upload --max-chunks-per-repo 200
```

### Multiple Repositories & Repos File

```bash
# Pass multiple repositories
python main.py tokio-rs/tokio serde-rs/serde reqwest/reqwest --hf-repo YOUR_USER/rust-core

# Or supply a text file with one repository per line
python main.py --repos-file repos.txt --hf-repo YOUR_USER/rust-collection
```

### Using Local or Alternative LLMs (Ollama, vLLM, OpenRouter)

```bash
python main.py BurntSushi/ripgrep \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --no-upload
```

### Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `repos` | - | One or more `owner/name` or GitHub URLs (up to 10) |
| `--repos-file` | - | Text file containing repositories (one per line) |
| `--model` | `gpt-4o-mini` | LLM model name |
| `--base-url` | OpenAI | Base URL for OpenAI-compatible endpoint |
| `--concurrency` | `8` | Parallel LLM generation requests |
| `--max-chunks-per-repo`| `None` | Cap sampled chunks per repository to control cost |
| `--max-chunk-chars` | `6000` | Maximum character length per code snippet |
| `--min-chunk-lines` | `3` | Minimum line count to consider a snippet |
| `--include-tests` | `False` | Include test files and test functions |
| `--no-upload` | `False` | Save files locally under `out/` without Hugging Face upload |
| `--private` | `False` | Create the Hugging Face dataset as private |

---

## 📂 Project Structure

```text
├── core.py              # Configuration dataclasses, logging, and constants
├── repo.py              # Git cloning, repository cleaning, and metadata parsing
├── loop.py              # LLM prompt synthesis and async generation loop
├── convert.py           # Chunking and Parquet / README dataset conversion
├── pip.py               # Hugging Face Hub dataset card & parquet uploader
├── main.py              # CLI entry point
├── ui.py                # Standalone HTTP console dashboard
├── run_colab_kaggle.sh  # Cloudflare tunnel launcher for notebooks
├── README_TEMPLATE.md   # Default Hugging Face dataset card template
└── requirements.txt     # Python dependencies
```

---

## 📄 License

Distributed under the Apache 2.0 License. See source repositories for original code licenses retained in metadata.
