## NewsQA: News Dataset for QA Generation
[![Build Status](https://img.shields.io/github/actions/workflow/status/faizan1234567/QALLM/build.yml)](https://github.com/faizan1234567/QALLM/actions)
[![License](https://img.shields.io/github/license/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/blob/main/LICENSE)
[![Language](https://img.shields.io/github/languages/top/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM)
[![Contributors](https://img.shields.io/github/contributors/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/faizan1234567/QALLM)](https://github.com/faizan1234567/QALLM/issues)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faizan1234567/QALLM/blob/main/notebooks/qa_generation/qa_generation.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/faizan1234567/QALLM/blob/main/notebooks/qa_generation/qa_generation.ipynb)
[![Open in Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/faizan1234567/QALLM/blob/main/notebooks/qa_generation/qa_generation.ipynb)



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

Our case study revealed that while Llama2 offers the best quality, it is slower compared to GPT models. ```T5-small```, though fast, has limitations in accuracy and duplication. Consequently, we used ```GPT-3.5 Turbo``` and ```GPT-4``` to generate a more substantial dataset.

This dataset is open-source and can be used for:

- Fine-tuning LLMs
- Evaluating model performance

Additionally, we have fine-tuned Tiny Llama on this dataset.

### QA Generated Dataset Examples

<table>
<tr><th>LLaMA2</th><th>T5-small</th></tr>
<tr><td>

| Question                                                 | Answer                                  |
|----------------------------------------------------------|-----------------------------------------|
| What is Pakistan's official name?                        | Islamic Republic of Pakistan.           |
| How many people live in Pakistan?                        | Over 241.5 million as of 2023.          |
| What is the capital of Pakistan?                         | Islamabad.                              |
| What is the largest city and financial center of Pakistan? | Karachi.                                |

</td><td>

| Question                                                 | Answer                                  |
|----------------------------------------------------------|-----------------------------------------|
| What is the capital city of Sindh?                       | Karachi                                 |
| What is the population of Karachi?                       | over 20 million                         |
| Where is Karachi located?                                | southern tip of the country along the Arabian Sea coast |
| What is the capital city of Pakistan?                    | Islamabad                               |

</td></tr>
</table>


<table>
<tr><th>GPT-3.5-Turbo</th><th>GPT-4</th></tr>
<tr><td>

| Question                                          | Answer                                          |
|---------------------------------------------------|-------------------------------------------------|
| What inspired the founding of LAPS?               | The first rescued animal, a pit bull named Lucky.|
| How many dogs are currently housed at LAPS?       | Nearly 300 dogs.                                |
| How many stray animals have been vaccinated by LAPS so far?| Over 5,000 stray animals.                    |
| How many dogs and cats have been neutered by LAPS?| More than 3,000 dogs and cats.                  |

</td><td>

| Question                                          | Answer                                          |
|---------------------------------------------------|-------------------------------------------------|
| What are monopolistic seed companies doing to consumers? | Charging heavy costs.                       |
| How are farmers being facilitated in operating tube wells? | By using solar energy.                     |
| What steps are proposed to materialize a green revolution in the country?| Direct fertiliser subsidy, quality seeds supply, and solar-powered tube-wells. |
| How would the mentioned steps impact productivity?| Productivity would triple in a couple of years. |

</td></tr>
</table>



```GPT3.5-Turbo``` and ```GPT4``` generates desired response. 
![alt text](https://github.com/faizan1234567/QALLM/blob/main/images/gradio_demo.PNG)
***Fig.*** Gradio demo using ```T5-small```

### Installation

```bash
 git clone https://github.com/faizan1234567/QALLM.git
 cd QALLM
```

Create  a virtual enviroment using python venv
```bash
python3 -m venv qa_llm
source qa_llm/bin/activate
```
alternatively, you can use anaconda package manager
```bash
conda create -n qa_llm python=3.8.10 -y
conda activate qa_llm
```

Now install all the required dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage
QA generation, make sure to read and understand the configs and replace appropriate values as required.
```bash
python create_alpaca_format_dataset.py --chunk_size 5000 --dataset <path>
```
and run QA generation 
```bash
python qa_generator.py --model T5-small --cfg cfg/qa_generator.yaml
```