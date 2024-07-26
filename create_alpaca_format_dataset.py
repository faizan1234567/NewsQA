from typing import Tuple, List, Union
from pathlib import Path
import json
import csv
import pandas as pd
import sys
import os
import argparse

class Chunker:

  def __init__(self, data_file: Path = None) -> None:
    self.data_file = data_file

  def read_dataset(self):
    try:
      df = pd.read_csv(self.data_file)
      return df
    except FileNotFoundError:
      print('File does not exist')
      sys.exit()

  # shuffle the data frame content inplace and reset the index
  def randomize_dataset(self):
    df = self.read_dataset()
    random_df = df.sample(frac=1).reset_index(drop=True)
    return random_df

  # chunk the dataset now
  def get_data_chunk(self, chunk_size: int, save: bool = True):
    random_data = self.randomize_dataset()
    chunked_data = random_data.iloc[:chunk_size]
    if save:
      os.makedirs("dataset/QA/chunked_data", exist_ok = True)
      save_path = f"dataset/QA/chunked_data/qa_chunked_data_{chunk_size}.csv"
      chunked_data.to_csv(save_path, index=False)
    return save_path

def create_dataset(chunked_data: Path):
  """
  Prepare the QA dataset in the Alpaca instruction fine-tunnig format
  -------------------------------------------------------------------

  Parameters
  ----------
  chunked_data: splitted qa pairs dataset with 1000, 2000, or any other size
  """
  alpaca_format_data = []

  # read a csv file and store question answer pairs in the dataset list
  with open(chunked_data, 'r', encoding='utf-8') as file:
      reader = csv.reader(file)
      next(reader)

      # in this case we don't have any input so keep it empty
      for row in reader:
          question, answer = row
          alpaca_format_data.append({
              "instruction": question,
              "input": "",
              "output": answer
          })

  # Write data to a json file with same name as the chunked file name
  save_data = chunked_data.split('/')[-1].split('.')[-2] + '.json'
  fine_tune_dataset_dir = "dataset/alpaca_format"
  os.makedirs(fine_tune_dataset_dir, exist_ok = True)
  with open(os.path.join(fine_tune_dataset_dir, save_data), 'w', encoding='utf-8') as file:
      json.dump(alpaca_format_data, file, ensure_ascii=False, indent=4)
  print("Dataset successfully converted and written to", save_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="command line args")
  parser.add_argument("--chunk_size", type = int, help = "dataset chunk size")
  parser.add_argument("--dataset", type = str, help = "path to the dataset (large)")
  args = parser.parse_args()
  
  # Chunked dataset will be saved to dataset QA dir
  chunker = Chunker(args.dataset)
  chunked_path = chunker.get_data_chunk(args.chunk_size)

  # now create alpaca like format dataset
  print("Creating Alpaca format dataset")
  create_dataset(chunked_data=chunked_path)
  print("Dataset sucessfully created!")