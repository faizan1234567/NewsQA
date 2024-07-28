"""
===========================================
Generate News QA dataset to finetune an LLM 

python qa_generator.py -h
===========================================
"""

import argparse
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import json
import sys
from pathlib import Path
import yaml
import torch
from cfg import from_dict
from utils import preprocess_text, post_process_text, process_all_data
from huggingface_hub import hf_hub_download
from lmqg import TransformersQG
import pandas as pd
from openai import OpenAI

# Configure logger (stream handler)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


# Suppress output
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def llama2(prompt):
    '''genarate text given prompt'''
    response = lcpp_llm(prompt=prompt, max_tokens=256, temperature=0.4, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=False)
    llm_completion = response["choices"][0]["text"]  # Retrieving generated text only

    return llm_completion

def qag_generator(text):
  '''
  generate question answer and store them in a csv file

  parameters
  ----------
  text: str
  '''
  question_answer_pairs = model.generate_qa(text)

  # Write to csv file
  file_path = 'qa_pairs_new_t5_small.csv'

  if not os.path.exists(file_path):
      with open(file_path, 'w', newline='', encoding='utf-8') as file:
          writer = csv.writer(file)
          writer.writerow(['Question', 'Answer'])

  # Append data
  with open(file_path, 'a', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      writer.writerows(question_answer_pairs)


def QAG(dataset, text_splitter):
  """
  run over the json file and store them in a csv

  parameters
  ----------
  dataset: str
  text_splitter

  return
  ------
  None
  """
  for data in dataset:
    # Don't process articles that have already been explored by llm
    if 'processed_article' in data.keys():
      if data['processed_article']:
        logger.info('News article already processed!')
    else:
      text = data['text']
      chunks_list = text_splitter.split_text(text)

      for text_chunk in chunks_list:
        # Generate QA
        qag_generator(text_chunk)
      data['processed_article'] = True
      with open('dawn_pakistan_processed_t5_small', 'w') as file:
        json.dump(dataset, file, indent = 4)

def qa_generator(text):
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

  prompt_template = prompt + f"\n\nUSER: {text} \n\nASSITANT:"
  generated_text = llama2(prompt_template)
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
  for data in dataset:
    # Don't process articles that have already been explored by llm
    if 'processed_article' in data.keys():
      if data['processed_article']:
        print('News article already processed!')
    else:
      text = data['text']
      chunks_list = text_splitter.split_text(text)
      for text_chunk in chunks_list:
        # Genrate qa
        qa_generator(text_chunk)
      # this flag helps identify the articles already processed by an llm
      # so we don't repeat it again.
      data['processed_article'] = True
      with open('dawn_pakistan_processed', 'w') as file:
        json.dump(dataset, file, indent = 4)



def gpt_turbo(text):
  """
  use openai gpt3.5 turbo-0.125 model for qa geneartion

  parameters
  ----------
  text: context (str)

  return
  ------
  respone: qa pairs
  """
  model =  "gpt-3.5-turbo-0125"

  system_requirements = "You are a helpful and honest assistant for QA pairs generation."

  # prompt
  prompt = f"""
  Please Generate Question Answer pairs from text below. The format of QA pairs should
  be question first followed by an answer. An example of the QA format is shown below:

  Question: your question generated from text?
  Answer: your answer generated from text.

  Please genearte 3 to 5 QA pairs from each text and keep your answer as concise as possible.

  here is text: {text}"""

  try:
    response = client.chat.completions.create(
                                              model=model,
                                              messages=[
                                                  {"role": "system", "content": system_requirements},
                                                  {"role": "user", "content": prompt}
                                              ]
                                              )
    return response.choices[0].message.content
  except Exception as e:
    return f"An error occurred: {e}"


def process_save_qa_data(gpt_completion):
  
  qa_list = gpt_completion.split('\n')

  qa_list_empty_removed = [item for item in qa_list if item]

  # keep only quetions and answer and remove anything else
  data = [item.split(":", 1)[1].strip() for item in qa_list_empty_removed]

  # create qa pairs
  if len(data) % 2 != 0:
      data.pop(-1)
      qa_pairs = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
  else:
        qa_pairs = [(data[i], data[i+1]) for i in range(0, len(data), 2)]

  file_path = 'qa_pairs_gpt35_turbo1.csv'

  if not os.path.exists(file_path):
      with open(file_path, 'w', newline='', encoding='utf-8') as file:
          writer = csv.writer(file)
          writer.writerow(['Question', 'Answer'])

  # append data
  with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(qa_pairs)



def generate_qa_gpt(dataset, text_splitter):
  """
  run over the json file and store them in a csv, and use gpt3.5 turbo for qa generation

  parameters
  ----------
  dataset: str
  text_splitter

  return
  ------
  None
  """
  for data in dataset:
    if 'processed_article' in data.keys():
      if data['processed_article']:
        logger.info('News article already processed!')
    else:
      # reterive text
      text = data['text']
      chunks_list = text_splitter.split_text(text)

      # iterate over each chunk
      for text_chunk in chunks_list:
        completion = gpt_turbo(text_chunk)
        if completion.startswith("An error occurred"):
          pass
        else:
          process_save_qa_data(completion)

        # this flag helps identify the articles already processed by an llm
        # so we don't repeat it again.
      data['processed_article'] = True
      logger.info('An article has been processed!!')
      
      with open('dawn_pakistan_processed_recent_gpt35', 'w') as file:
        json.dump(dataset, file, indent = 4)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Command Line arguments")
  parser.add_argument(
    "--model",
    type=str,
    choices=["GPT3.5-Turbo", "GPT-4", "Llama2", "T5-small"],
    default="GPT3.5-Turbo",
    help="name of the model"
   )
  
  parser.add_argument("--cfg", 
                      type= str, 
                      choices = ["cfg/qa_generator.yaml", "cfg/fine_tuning.yaml"], 
                      default= "cfg/qa_generator.yaml", 
                      help = "configuration file")
  
  opt = parser.parse_args()

  # check GPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
     logger.info("WARNING: Running with CPU could be slow")

  # Init config
  cfg = yaml.safe_load(Path(opt.cfg).open('r'))
  cfg = from_dict(cfg)  # convert dict
  

  # load dataset
  with open(cfg.sample_dataset) as f:
    dataset = json.load(f)
  dataset = process_all_data(dataset)

  # Define text splitter
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=cfg.chunk_size, 
      chunk_overlap=cfg.chunk_overlap,
      length_function=len,
      is_separator_regex=False,
  )
   # Sample text
  with open(cfg.sample_text, "r", encoding= "utf-8") as file:
    text = file.read()

  paragraphs = text.split("\n\n")
  text = paragraphs[0] if len(paragraphs) > 0 else ""
  
  # Dont print sample process
  print_response = cfg.print_response

  if opt.model == "Llama2":
    logger.info("Loading Llama2 for QA generation")
    logger.info("WARNING: Llama2 maybe slow")
    # Define model
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
    model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

    from llama_cpp import Llama
    logger.info("Download the model")
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    with SuppressOutput():
      lcpp_llm = Llama(
          model_path=model_path,
          n_threads=2, # CPU cores
          n_batch=512,
          n_gpu_layers=32
          )
    logger.info(f"Number of GPU layers {lcpp_llm.params.n_gpu_layers}")

   
    # prompt template
    prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant for QA data acquisition.
    Generate question and answer pairs using the infromation from the USER text below.
    Generate your own questions from the context below and then generate answers from the text for each question
    you generated.


    USER: {text}

    ASSISTANT:
    '''
    
    # Invoke the Llama2 for generation if print_response is true
    if print_response:
      response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95,
                    repeat_penalty=1.2, top_k=150,
                    echo=False)
      
      # sample response 
      print(response["choices"][0]["text"])


    logger.info("Generating QA using Llama2")
    create_qa(dataset, text_splitter)
  
  elif opt.model == "T5-small":
    model = TransformersQG(language='en', model='lmqg/t5-base-squad-qag')
    QAG(dataset, text_splitter)

  elif opt.model == "GPT-3.5Turbo":
    system_requirements = "You are a helpful and honest assistant for QA pairs generation."
    # prompt
    prompt = f"""
    Please Generate Question Answer pairs from text below. The format of QA pairs should
    be question first followed by an answer. An example of the QA format is shown below:

    Question: your question generated from text?
    Answer: your answer generated from text.

    Please genearte 3 to 5 QA pairs from each text and keep your answer as concise as possible.

    here is text: {text}"""

    model = cfg.gpt_model
    # replace by your api key here
    api_key = cfg.gpt_api_key
    if print_response:
      client = OpenAI(api_key=api_key)

      def GPT35Turbo(prompt, model, system_requirements):
          try:
              response = client.chat.completions.create(
                  model=model,
                  messages=[
                      {"role": "system", "content": system_requirements},
                      {"role": "user", "content": prompt}
                  ]
              )
              return response.choices[0].message.content
          except Exception as e:
              return f"An error occurred: {e}"
      if print_response:
        response = GPT35Turbo(prompt, model, system_requirements)
        print(response)
    generate_qa_gpt(dataset, text_splitter)
  


    









