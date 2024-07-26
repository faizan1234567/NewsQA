
# News Dataset for QA Generation

This repository contains a large dataset of news articles scraped from various Pakistani news websites. The dataset covers diverse categories including politics, sports, fashion & style, international news, domestic affairs, and science & technology.
Data Collection and QA Generation

We used several large language models (LLMs) for generating question-answer pairs from the scraped news articles:

    **Llama2**: Generates high-quality question-answer pairs but is relatively slow.
    **T5-small**: Fast but less accurate, often producing duplicate question-answer pairs.
    **GPT-3.5 Turbo and GPT-4**: We found these models to be effective for generating high-quality question-answer pairs efficiently.

## Findings and Dataset

Our case study revealed that while Llama2 offers the best quality, it is slower compared to GPT models. T5-small, though fast, has limitations in accuracy and duplication. Consequently, we opted to use GPT-3.5 Turbo and GPT-4 for generating a more substantial dataset.

This dataset is open-source and can be used for fine-tuning LLMs and evaluating model performance. We have also fine-tuned Tiny Llama on this dataset.