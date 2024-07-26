## NewsQA: News Dataset for QA Generation
[![Build Status](https://img.shields.io/github/actions/workflow/status/faizan1234567/QALLM/build.yml)](https://github.com/faizan1234567/QALLM/actions)
[![License](https://img.shields.io/github/license/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/release/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/releases)
[![Language](https://img.shields.io/github/languages/top/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM)
[![Contributors](https://img.shields.io/github/contributors/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/issues)


This repository contains a large dataset of news articles scraped from various Pakistani news websites. The dataset covers diverse categories including:

- Politics
- Sports
- Fashion & Style
- International News
- Domestic Affairs
- Science & Technology

### Data Collection and QA Generation

We evaluated several large language models (LLMs) for generating question-answer pairs from the scraped news articles:

- **Llama2**: Generates high-quality question-answer pairs but is relatively slow.
- **T5-small**: Fast but less accurate, often producing duplicate question-answer pairs.
- **GPT-3.5 Turbo and GPT-4**: Effective for generating high-quality question-answer pairs efficiently.

### Findings and Dataset

Our case study revealed that while Llama2 offers the best quality, it is slower compared to GPT models. T5-small, though fast, has limitations in accuracy and duplication. Consequently, we used GPT-3.5 Turbo and GPT-4 to generate a more substantial dataset.

This dataset is open-source and can be used for:

- Fine-tuning LLMs
- Evaluating model performance

Additionally, we have fine-tuned Tiny Llama on this dataset.

