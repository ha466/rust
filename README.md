# Rustset

Rustset builds a question-to-code dataset from public Rust GitHub repositories. It clones each source, keeps eligible Rust code, asks an OpenAI-compatible model to write developer prompts for the code, writes a Parquet file and Hugging Face dataset card, and can upload both to the Hub.

## Run in Google Colab or Kaggle

Enable internet access for the notebook, then run the matching commands. Replace the repository URL and email with your own:

**Google Colab**

```python
!git clone https://github.com/OWNER/REPOSITORY.git rustset
%cd rustset
!CLOUDFLARE_ALLOWED_EMAIL="you@example.com" bash run_colab_kaggle.sh
```

**Kaggle**

```python
!git clone https://github.com/OWNER/REPOSITORY.git /kaggle/working/rustset
%cd /kaggle/working/rustset
!CLOUDFLARE_ALLOWED_EMAIL="you@example.com" bash run_colab_kaggle.sh
```

The script installs Python requirements and `cloudflared`, starts the dashboard, then opens a free Cloudflare Quick Tunnel with email verification. Open the temporary `trycloudflare.com` address printed in notebook output and enter the allowed email. Keep the cell running while you use the dashboard; stop it to close the tunnel.

Quick Tunnels are temporary and intended for development. Only the configured email can pass Cloudflare's check. In Kaggle, enable Internet in notebook settings. If the notebook environment blocks package downloads or Cloudflare connections, use the CLI locally instead.

## Dashboard settings

- **GitHub repositories:** one `owner/name` or GitHub URL per line, up to 10.
- **API key and endpoint:** use an OpenAI API key, or the API key and base URL for any OpenAI-compatible endpoint. The model defaults to `gpt-4o-mini`.
- **Hugging Face token and dataset repo:** both are needed to upload. Leave the dataset repo blank to save files locally under `out/`.
- **README template:** edit the Hugging Face dataset card before starting. `README_TEMPLATE.md` is the default. Supported tokens are `{{dataset_repo}}`, `{{model}}`, `{{row_count}}`, `{{repo_count}}`, `{{license}}`, `{{size_category}}`, `{{source_repositories}}`, `{{rows_by_kind}}`, and `{{example}}`.

Credentials entered in the dashboard stay in the running Python process and are not saved to a file. Cloudflare email verification protects the dashboard; stop the notebook cell after the run.

## Run from a terminal

```bash
pip install -r requirements.txt
python main.py owner/repo --api-key YOUR_API_KEY --hf-repo YOUR_USER/dataset --hf-token YOUR_HF_TOKEN
```

Set `--base-url` for a compatible API endpoint. Add `--no-upload` to write locally. Use `python main.py --help` to see chunking and cost-control options.

Outputs go to `out/`; resumable work files go to `work/`. Both are excluded from Git.
