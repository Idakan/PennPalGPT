Notebooks:

Please access TrainingDataSmall.zip from [here]([url](https://drive.google.com/file/d/17YDixaHlEz4GYiLQfAQnhJeds3-UBvxp/view?usp=drive_link)). 

feature_generation.ipynb - In this notebook, we use a gpt-3.5 model to help generate features and use few-shot prompting with 3 examples of articles with extracted topics discussed. We pass the question 'What are key features of {professor's name} work?' for every professor to get a large set of possible features that captures every professor. We tried this with a couple different types of prompts but found the few shot prompt the most successful. Upon receiving these results, we created a dataframe 'professor_research_results.csv' with 'professor' and 'features' columns. We created 93 features from this list, coming up with a combination of broad and niche features. 

To run this notebook, please upload the pyproject.toml and TrainingDataSmall.zip file and run using gpus.

grouping_professors.ipynb - In this notebook, we use our dataset 'data_professors.csv' to analyze how similar our professors are and used K-Means to find 6 groups that we later used in our score.py script that allows more context into how close our answers are to being correct.

To run this notebook, please upload 'data_professors.csv'

feature_transformation_for_test.ipynb - In this notebook, we create a gpt-4 model and a prompt that instructs the model return 0 and 1 for all of our features. We then create a matrix for all of our question's features and build both a Random Forest classifier, SVC, and KNN model. Both models are trained on the professor data and we predict a professor from our question matrices. The files we used to run this are '530_project_test_dev.xlxs', our dev data, and data_professors.csv. Using this notebook, we have output 'question_matrices.csv', which consist our question matrices, 'results_random_forest.csv', which consists our random forest results, 'results_svc', which consists our SVC results, 'results_knn.csv', which consists our KNN results. Note that our SVC and KNN results consist a list of professors, which is what we initially proposed for our model to return.

Python files:

score.py - very similar to score.py in Milestone 2 except we updated our prompt so we can better evaluate the quality of our answer. 
