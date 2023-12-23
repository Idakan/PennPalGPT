In the outputs folder, we have 4 files, knn_results_score_dev.csv, knn_results_score_test.csv, knn_results_dev.csv, and knn_results_dev.csv.

knn_results_dev.csv and knn_results_dev.csv consist the chatbot output obtained from machine learning.

knn_results_score_dev.csv and knn_results_score_dev.csv consist the scores of each chatbot answer obtained from the evaluate script.

Use score.py and the following command line prompts below to obtain the results:

To get dev scores: python score.py knn_results_score_dev.csv

To get test scores: python score.py knn_results_score_test.csv

Note: The file must consist columns 'chatbot_answer' and 'golden_standard_answer'.
