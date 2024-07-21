import re
import os
import csv



def preprocess_text(text):
  """
  preporcess the text and remove special character and urls if there are any

  Parameters
  ----------
  text: str

  Return
  ------
  cleaned_text: str
  """
  cleaned_text =  re.sub(r'[^\w\s\.,!?]', '', text)

  # remove urls and replace it with none space
  cleaned_text = re.sub(r"http\S+", "", cleaned_text)
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
  return cleaned_text.strip()


def process_all_data(dataset):
  """
  process all the data in the file that we just read in a json file

  parameters
  -----------
  dataset: list (raw)

  return
  ------
  dataset: list (processed)
  """
  for data in dataset:
    text = data["text"]
    processed_text = preprocess_text(text)
    data["text"] = processed_text
  return dataset



def find_questions_indices(data):
  '''
  find question "?" identifiers index in the list

  parameters
  ----------
  data: list

  return:
  ------
  first_question_index: int
  last_question_index: int
  '''
  first_question_index = None
  last_question_index = None

  # Iterate through the list
  for i, item in enumerate(data):
      if item.endswith('?') or '?' in item:
          # If first_question_index is None, set it to the current index
          if first_question_index is None:
              first_question_index = i
          # Update last_question_index to the current index
          last_question_index = i
  return (first_question_index, last_question_index)


def has_consecutive_ones(lst):
  '''
  find if the completion contains all the questions when no answer generated.

  parameters
  ----------
  lst: list

  return
  ------
  var: bool
  '''
  for i in range(len(lst) - 1):
      if lst[i] == 1 and lst[i + 1] == 1:
          return True
  return False

def post_process_text(llm_completion):
  '''
  process the llm completion to generate question answer pairs and save them in a csv file

  parameters
  ----------
  llm_completion: str

  return
  ------
  None
  '''
  qa_list = llm_completion.split('\n') # separte each line of completion qa

  # remove first entry as it is just the model response not actual qa
  qa_list.pop(0)
  # remove empty items
  qa_list = [item for item in qa_list if item]
  # find first and last question indices
  first_question_ind, last_question_ind = find_questions_indices(qa_list)
  # get disired qa pairs
  qa_data = qa_list[first_question_ind: last_question_ind+2]
  # keep only quetions and answer and remove anything else
  data = [item[item.rfind(":") + 1:].strip() if ":" in  item
        else item[item.rfind(".") + 1:].strip() if "." in item
        else item[item.rfind(")") + 1:].strip() if ")" in item
        else item.strip() for item in qa_data]

  # checking if model is generating questions only then don't save in csv
  data_items = [1 if '?' in item else 0 for item in data]

  only_questions = has_consecutive_ones(data_items)
  if only_questions:
    # don't store anything in this case
    pass
  else:
    # create qa pairs
    print('writing output to csv..')
    # should be even number of pairs
    if len(data) % 2 != 0:
      data.pop(-1)
      qa_pairs = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
    else:
      qa_pairs = [(data[i], data[i+1]) for i in range(0, len(data), 2)]

    # write to csv file
    file_path = 'qa_pairs.csv'

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(['Question', 'Answer'])

    # append data
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(qa_pairs)

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
  # complte prompt: text ingested in the prompt
  prompt_template = prompt + f"\n\nUSER: {text} \n\nASSITANT:"
  # invoke llm (llama2)
  generated_text = llama2(prompt_template)
  # logg model completion for now
  logger.info(generated_text)
  # process the completion and write the qa pairs in a csv file
  post_process_text(generated_text)