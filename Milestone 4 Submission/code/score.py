import openai
from openai import OpenAI
from time import sleep
import os
import collections
import pandas as pd
import sys
import time
import numpy as np

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

  Group 1: Hamed Hassani, Sanjeev Khanna, Michael Kearns, Robin Pemantle, Aaron Roth, Jacob Gardner, Shivani Agarwal,
  Anindya De, Eric Weingarten, Rakesh Vohra, Weijie Su
  Group 2: Lingjie Liu, Eric Eaton, Nadia Figueroa, Dinesh Jayaraman, Mingmin Zhao, Jianbo Shi, Dan Roth, Andrew Head, 
  Rajeev Alur, Justin Gottschlich, Charles Yang, Michael Posa, Kostas Daniilidis, Alejandro Ribeiro, Cynthia Sung,
  Camillo Taylor, M. Ani Hsieh, Vijay Kumar, Daniel E. Koditschek
  Group 3: Linh Thi Xuan Phan, Jonathan Smith, Ryan Marcus, Tal Rabin, Vincent Liu, Mayur Naik,
  Benjamin C. Pierce, Susan Davidson, Andreas Haeberlen, Zachary Ives, Sebastian Angel, Lin Thi Xuan Phan
  Group 4: Rajiv Gandhi, Brett Hemenway, Harry Smith, Adam David Mally,
  Andre Scedrov, Bong Ho Kim, Jérémie O. Lumbroso, Nikolai Matni, Travis Q. McGaha, Pratyush Mishra,
  Shirin Saeedi Bidokhti, Boon Thau Loo, Jing Li, Benjamin Lee, Stephen Lane, Joe Devietti, Pratik Chaudhari,
  Jean Gallier, Thomas Farmer, Val B. Tannen, Christopher S. Yoo, Scott Weinstein, Stephanie Weirich, Steven Zdancemi
  Group 5: Konrad Kording, Qi Long,  Victor M. Preciado, Insup Lee, Yoseph Barash
  Group 6: Daniel Hashimoto, Junhyong Kim, Harvey Rubin, Joshua B. Plotkin, Mark L. Liberman, Kevin B. Johnson,
  Sampath K. Kannan, James Gee, Norman I. Badler, Li-San Wang, Oleg Sokolsky, Rene Vidal
  Group 7: Eric Fouh, Gushu Li, Yasmin Kafai, Ryan Baker, Swapneel Seth
  Group 8: Osbert Bastani, Surbi Goel, Chris Callison-Burch, Mark Yatskar, Eric Wong, Danaë Metaxa, 
  Damon Centola, Sharath Chandra Guntuku, Daniel J Hopkins, Duncan Watts, Lyle Ungar

  Check the criteria below to assign a score:
   - Return a 1 if a chatbot answer is not returned or the chatbot answer is not a professor and the golden standard answer is.
   - Return 2 if NONE of the names in the chatbot answer are in the same group (see the groups above) as the golden standard answer.
   - Return 3 if one of the names in the chatbot answer are professors that are in the same group (see the groups above) as the golden standard answer
   - Return 4 if more than one of the names in the chatbot answer are professors that are in the same group (see the groups above) as the golden standard answer
   - Return 5 if one of the chatbot answer names is the same professor as the golden standard answer OR the chatbot and golden standard answers are the same

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
  df = pd.read_csv(file)
  for chatbot_answer, golden_answer in zip(chatbot, golden):
    time.sleep(3)
    score = get_score(chatbot_answer, golden_answer)
    scores.append(score)
    print('CHATBOT ANSWER: {} GOLDEN ANSWER: {}  SCORE: {} / 5'.format(chatbot_answer, golden_answer, score))
  total_score = get_total_score(scores)
  df['score'] = np.array(scores)
  print('TOTAL SCORE (0-1):', round(total_score, 4))
  df.to_csv('knn_results_scores_{}.csv'.format(round(total_score, 4)))


#passes csv file to evaluate()
evaluate(sys.argv[1])