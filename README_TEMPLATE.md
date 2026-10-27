---
license: {{license}}
task_categories:
- text-generation
language:
- en
tags:
- rust
- code
- instruction-tuning
- synthetic
size_categories:
- {{size_category}}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Rust code instruction dataset

{{row_count}} pairs of **(question, real Rust code)** built from {{repo_count}} open-source GitHub repositories.

Each `answer` is original Rust source. Each `question` was written by `{{model}}` from the code context; the model does not generate the answer.

## Source repositories

| repository | commit | license | files | chunks | rows |
|---|---|---|---:|---:|---:|
{{source_repositories}}

## Rows by kind

| kind | rows |
|---|---:|
{{rows_by_kind}}

{{example}}
## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{{dataset_repo}}", split="train")
print(dataset[0]["question"])
print(dataset[0]["answer"])
```

## License and attribution

Code in `answer` remains under the license of the source repository shown in the `license` and `repo_url` columns. Follow that license when redistributing or training on the data. The questions are synthetic; follow the terms of the model provider used to create them.
