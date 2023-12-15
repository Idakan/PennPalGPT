import openai
from openai import OpenAI
from time import sleep
import os
import collections
import pandas as pd
import sys
import time

OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_prompt(chatbot_answer, golden_answer):
  """
  Consists prompt and formats it to fit chatbot and golden answers in this iteration

  Inputs: 
    chatbot_answer
    golden_answer

  Returns: formatted prompt
  """
  TASK_PROMPT = """
  Your task to return a score that ranges from 1 to 5 (where 1 is the lowest and 5 is the highest score)
  based on how close the chatbot 
  and golden standard answer are in semantic meaning and if they give the same answer.

  Group 1: Aaron Roth, Chris Callison-Burch, Osbert Bastani, Mark Yatskar
  Group 2: Ryan Marcus, Mayur Naik, Joshua B. Plotkin, Dan Roth, Mark L. Liberman, Robin Pemantle, Danaë Metaxa, Victor M. Preciado,
  Eric Fouh, Gushu Li, Pratik Chaudhari, Damon Centola, Sharath Chandra Guntuku, Andreas Haeberlen, Andrew Head,
  Brett Hemenway, Daniel J Hopkins, Yasmin Kafai, Sebastian Angel, Shivani Agarwal, Jean Gallier, Andre DeHon, Anindya De,
  Ryan Baker, Christopher S. Yoo
  Group 3: Bong Ho Kim, Adam David Mally, Andre Scedrov, Tal Rabin, Vincent Liu, Jérémie O. Lumbroso, Nikolai Matni,
  Alejandro Ribeiro, Travis Q. McGaha, Benjamin C. Pierce, Pratyush Mishra, Shirin Saeedi Bidokhti, Boon Thau Loo,
  Rajiv Gandhi, Jing Li, Insup Lee, Benjamin Lee, Stephen Lane, Vijay Kumar, Joe Devietti, Susan Davidson,
  Sanjeev Khanna, Thomas Farmer, Val B. Tannen, Charles Yang, Duncan Watts, Eric Weingarten, Harry Smith, Jonathan Smith
  Oleg Sokolsky, Rakesh Vohra, Scott Weinstein, Stephanie Weirich, Steven Zdancemic, Swapneel Seth
  Group 4: Junhyong Kim, Qi Long, Harvey Rubin, Yoseph Barash, Kevin B. Johnson, Sampath K. Kannan, James Gee, Li-San Wang
  Rene Vidal
  Group 5: Daniel E. Koditschek, Linh Thi Xuan Phan, Rahul Mangharam, George J. Pappas, Michael Posa, Nadia Figueroa
  Daniel Hashimoto, M. Ani Hsieh, Kostas Daniilidis, Norman I. Badler, Mingmin Zhao, Mark Yim, Camillo Taylor, Cynthia Sung

  Check the criteria below to assign a score:
   - Return a 1 if a chatbot answer is not returned.
   - Return 2 if the chatbot answer is not a professor.
   - Return 3 if the chatbot answer is a professor that is a UPenn professor but are not in the same group (see the groups above) as the golden standard answer
   - Return 4 if the chatbot answer is a professor that is in the same group (see the groups above) as the golden standard answer
   - Return 5 if the chatbot answer is the same professor as the golden standard answer

   The chatbot answer is {input_1} and the golden standard answer is {input_2}. 
   You must return a number and nothing else between 1 and 5.
   The returned score is:
  """
  prompt = TASK_PROMPT.replace("{input_1}", chatbot_answer).replace("{input_2}", golden_answer)
  return prompt

def model(chatbot_answer, golden_answer):
  """
  Calls our GPT-4 model and returns its response

  Inputs: 
        chatbot_answer
        golden_answer
    
  Returns: single digit response between 1 and 5
  """
  client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
  prompt = get_prompt(chatbot_answer, golden_answer)
  chat_completion = client.chat.completions.create(
                      messages=[{"role": "user", "content": prompt}],
                      model="gpt-4-0613")
  response = dict(dict(dict(chat_completion)['choices'][0])['message'])['content']
  return response

def get_score(chatbot_answer, golden_answer):
  """
  Calls the model to retrieve the score
  
  Inputs:
        chatbot_answer
        golden_answer

  Returns: score as an integer
  """
  score = model(chatbot_answer, golden_answer)
  return int(score)

def get_prior_probs(scores_list):
  """
  Finds distribution of scores during the trial

  Inputs:
        scores_list - list of scores for all chatbot answers in the trial

  Returns: a dictionary of each possible score 1-5 and the probability of each occuring 
  """
  score_freq = collections.Counter(scores_list)
  for num, count in score_freq.items():
    score_freq[num] = round(count/len(scores_list), 2)
  return score_freq

def get_total_score(scores_list):
  """
  Calculates the overall score for the model according to this trial's scores

  Input:
        scores_list - list of scores for all chatbot answers in the trial

  Output: the final score represented as a sum of each score and its probability divided by the highest possible score 5
  """
  priors = get_prior_probs(scores_list)
  total_score = 0
  for score in range(1,6):
    if score in priors:
      total_score += (score * priors[score])
  return total_score / 5

def read_file(file):
  """
  Reads the inputted file

  Input: 
        file - csv file with 'chatbot_answer' and 'golden_standard_answer' columns

  Output: chatbot and golden answer list of strings
  """
  chatbot_answer = pd.read_csv(file)['chatbot_answer'].tolist()
  golden_standard_answer = pd.read_csv(file)['golden_standard_answer'].tolist()
  return chatbot_answer, golden_standard_answer

def evaluate(file):
  """
  Iteratively prints the score of each answer and the final overall score

  Inputs:
        file - csv file with 'chatbot_answer' and 'golden_standard_answer' columns

  Output: none
  """
  scores = []
  print('EVALUATION')
  chatbot, golden = read_file(file)
  for chatbot_answer, golden_answer in zip(chatbot, golden):
    time.sleep(3)
    score = get_score(chatbot_answer, golden_answer)
    scores.append(score)
    print('CHATBOT ANSWER: {} GOLDEN ANSWER: {}  SCORE: {} / 5'.format(chatbot_answer, golden_answer, score))
  total_score = get_total_score(scores)
  print('TOTAL SCORE (0-1):', round(total_score, 4))

#passes csv file to evaluate()
evaluate(sys.argv[1])