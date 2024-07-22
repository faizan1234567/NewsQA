import argparse
import os
import csv
import json
from utils import preprocess_text, post_process_text, process_all_data
from huggingface_hub import hf_hub_download

# logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


def qa_generator(llama2, text):
  '''
  takes the paragraph text and generate model completion and then post process the completion
  to generate QA data

  parameters
  ----------
  text: str

  return
  ------
  None
  '''

  # prompt
  prompt=f'''SYSTEM: You are a helpful, respectful and honest assistant for QA data acquisition.
    Generate question and answer pairs using the infromation from the USER text below.
    Generate your own questions from the context below and then generate answers from the text for each question
    you generated.
  '''
  # complte prompt: text ingested in the prompt
  prompt_template = prompt + f"\n\nUSER: {text} \n\nASSITANT:"
  # invoke llm (llama2)
  generated_text = llama2(prompt_template)
  # logg model completion for now
  logger.info(generated_text)
  # process the completion and write the qa pairs in a csv file
  post_process_text(generated_text)

def create_qa(dataset, text_splitter):
  '''
  run the llm over the json file, and store the respone in a csv file
  if the articles that are published should be marked so they don't be explored again

  parameters
  ----------
  dataset: list[dict]
  text_splitter: Module from langchain

  return:
  ------
  None
  '''
  # iterate over all the json entries
  for data in dataset:
    # don't process articles that have already been explored by llm
    if 'processed_article' in data.keys():
      if data['processed_article']:
        print('News article already processed!')
    else:
      # reterive text
      text = data['text']
      # chunk the text as the text is bit longer in each article
      # longer text takes time and llm context window can't process all at once becasue of limited context size for some llm
      # split into 500 chars with 0 overlap
      chunks_list = text_splitter.split_text(text)

# llama2 llm
def llama2(prompt):
    # language model processing logic
    response = lcpp_llm(prompt=prompt, max_tokens=256, temperature=0.4, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=False)
    llm_completion = response["choices"][0]["text"]  # Retrieving generated text only

    return llm_completion

def main(args):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.parse_args("--model", type = str, default = "GPT3.5-Turbo", help= "name of the model")
  opt = parser.parse_args()

  if opt.model == "Llama2":
    logger.info("Loading Llama2 for QA generation")
    logger.info("WARNING: Llama2 maybe slow")
    # Define model
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
    model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

    from llama_cpp import Llama
    logger.info("Download the model")
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    lcpp_llm = None
    lcpp_llm = Llama(
        model_path=model_path,
        n_threads=2, # CPU cores
        n_batch=512,
        n_gpu_layers=32
        )
    logger.info(f"Number of GPU layers {lcpp_llm.params.n_gpu_layers}")

    # reterive the text
    with open("text_sample.txt", "r", encoding= "utf-8") as file:
      text = file.read()
    
    paragraphs = text.split("\n\n")
    text = paragraphs[0] if len(paragraphs) > 0 else ""

    # prompt template
    prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant for QA data acquisition.
    Generate question and answer pairs using the infromation from the USER text below.
    Generate your own questions from the context below and then generate answers from the text for each question
    you generated.


    USER: {text}

    ASSISTANT:
    '''
    
    # Invoke the Llama2 for generation
    response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=False)
    
    # sample response 
    print(response["choices"][0]["text"])

    # Define text splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    # load dataset
    preprocessed_data = 'dawn_pakistan.json'
    with open(preprocessed_data) as f:
      dataset = json.load(f)
    dataset = process_all_data(dataset)
    
    logger.info("Generating QA using Llama2")
    create_qa(dataset, text_splitter)

    









