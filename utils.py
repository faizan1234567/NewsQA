import re
import os
import csv

from datasets import load_metric
from transformers import TrainerCallback, TrainerState, TrainerControl
import os
import math
import json


# Load evaluation metrics
rouge = load_metric("rouge")
bleu = load_metric("bleu")


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

def get_response_text(text):
  """
  This function extracts the text within the ### Response: section, stopping at the next section marker (###).

  Args:
      text: The text containing the response section.

  Returns:
      The extracted response text, or an empty string if no response is found.
  """
  lines = text.splitlines()
  response_start = None

  for i, line in enumerate(lines):
    if line.startswith("### Response:"):
      response_start = i + 1
      break

  if response_start is not None:
    # Find the next line that starts with "#" (indicating the end of Response)
    for j in range(response_start, len(lines)):
      if lines[j].startswith("###"):
        response_end = j
        break
      else:
        response_end = len(lines)  # Set end to last line if no next section marker

    # Extract the response text between the start and end lines.
    return "\n".join(lines[response_start:response_end])
  else:
    # No response section found.
    return ""


def evaluate_model(model, tokenizer, test_dataset, QA_prompt):
    model.eval()
    predictions = []
    references = []

    for example in test_dataset:
        instruction = example["instruction"]
        reference = example["output"]

        inputs = tokenizer([
        QA_prompt.format(
        f"{instruction}", # instruction
        "")
          ], return_tensors = "pt").to("cuda")


        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True, pad_token_id=tokenizer.eos_token_id)

        prediction = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0] #tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])[0]
        prediction = get_response_text(prediction)
        predictions.append(prediction)
        references.append(reference)
    # calculate scores now
    rouge_result = rouge.compute(predictions=predictions, references=references)
    bleu_result = bleu.compute(predictions=[pred.split() for pred in predictions],
                               references=[[ref.split()] for ref in references])


    results = {
        "rouge": rouge_result,
        "bleu": bleu_result
    }

    return results


class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, examples_interval,  test_dataset):
        self.examples_interval = examples_interval  # Number of examples between checkpoints
        self.test_dataset = test_dataset
        # self.tokenizer = tokenizer
        self.steps_to_save = []  # To be calculated based on batch size and accumulation steps
        self.results = []

    def on_train_begin(self, args, state, control, **kwargs):
        # Calculate the steps at which to save checkpoints based on examples
        max_examples = state.max_steps * (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        self.steps_to_save = [math.ceil(i / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
                              for i in range(self.examples_interval, max_examples + 1, self.examples_interval)]

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.steps_to_save:
            control.should_save = True  # Save checkpoint at this step
        else:
            control.should_save = False  # Do not save checkpoint at this step

        return control

    def on_save(self, args, state, control, **kwargs):
        # Rename the checkpoint directory after saving
        step = state.global_step
        if step in self.steps_to_save:
            # Calculate the example count based on the current step
            example_count = step * (args.per_device_train_batch_size * args.gradient_accumulation_steps)
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            new_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{example_count}")

            if os.path.exists(checkpoint_dir):
                os.rename(checkpoint_dir, new_checkpoint_dir)
                print(f"Checkpoint saved and renamed to: {new_checkpoint_dir}")

                # Evaluate the current state of the model
                results = evaluate_model(kwargs['model'], kwargs['tokenizer'], self.test_dataset)
                print(f"Evaluation results for checkpoint-{example_count}: {results}")

                # Store results with checkpoint name
                self.results.append({
                    "checkpoint": f"checkpoint-{example_count}",
                    "results": results
                })

    def on_train_end(self, args, state, control, **kwargs):
        # Save results to a file or return them as needed
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Saved evaluation results to {results_file}")

