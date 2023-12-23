In the code folder, we have 5 files that we used for this Milestone. Please see the summaries below of what each file does:

feature_importance_analysis_and_eval.ipynb : analyzes features from results in Milestone 3 and includes results for Random Forest Feature Importance and KNN Feature Importance. To run, upload data_professors.csv, question_matrices.csv, results_random_forest.csv, and dev.xlsx to Google Colab directory which are in the data_for_feature_analysis folder in data


feature_generation.ipynb : generates features using GPT-4. As a result, we obtain the generated_features data and use it to create data_professors.csv in the generated_features folder in data. Please upload the TrainingDataSmall.zip folder to run the notebook


feature_transformation.ipynb : generates matrices for questions using GPT-4 and uses KNN to find output professor list.  This outputs results_knn_dev.csv and results_knn_test.csv in the output folder. Please upload dev.csv, data_professors.csv, and test.csv to run the notebook.

grouping_professors.ipynb : uses K-means++ to group professors. Please upload data_professors.csv.

score.py : our evaluation script which outputs knn_results_score_test.csv and knn_results_score_dev.csv in the output folder, which consist the score for each chatbot answer.

To run score.py use the following command in a terminal: python score.py <file>

Note: The file must consist columns 'chatbot_answer' and 'golden_standard_answer'.
