<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

# Goodreads-Sentiment-Analysis

This respository contains the code and reports completed as a part of the Weights and Biases (W&B) course "MLOps: Effective Model Development", see <a href="https://www.wandb.courses/courses/effective-mlops-model-development">here</a>. This course is a great introduction to the Weights and Biases ecosystem and clearly explains how to use the tools offered by W&B to track machine learning experiments, perform hyperparameter sweeps, and summarize results using W&B reports. In particular, the course explains how to use W&B artifacts to save and version both datasets and models.

As part of this course, I used the Huggingface transformers library and Weights and Biases to perform sentiment analysis on Goodreads reviews. Goodreads.com is a website where people can write reviews about books they have read and rate the book on a scale of 0-5. 
In this project I used encoder-only transformers in the BERT family to use the text of a review to predict the rating given by the user.
As a part of this course I wrote the following three reports: <a href="https://api.wandb.ai/links/dmeltzer/ilnx2o0v">Report 1</a>, <a href="https://api.wandb.ai/links/dmeltzer/s840cljt">Report 2</a>, and <a href="https://wandb.ai/dmeltzer/mlops-course-assgn3/reports/Goodreads-Reviews-Week-3--VmlldzozNzYxODkz">Report 3</a>.
In the next section I will give a brief summary of these reports.


## <a href="https://api.wandb.ai/links/dmeltzer/ilnx2o0v">Week 1</a>

In the first report I performed exploratory data analysis (EDA) on the Kaggle Goodreads dataset and trained two transformer models on multi-classification problem, DistilBERT and BERT-tiny. 
In total, the dataset consisted of 900k reviews of 25,474 different books written by 12,188 different reviewers.
To clean the dataset, I first lower-cased all the reviews, removed newline characters and trailing whitespaces, and removed all duplicate reviews.
In addition, I found the original dataset was imbalanced: higher ratings were much more frequent than lower ratings.
To fix this, and shrink the dataset to a more manageable size, I downsampled the dataset so that all ratings appeared the same number of times.
After removing this imbalance, I split the data into train, validation, and test sets such that each split contained different books.
The two transformer models were trained on the training set and were monitored by periodically evaluating their accuracy on the validation set.
The best performing DistilBERT model had an accuracy of 60% on the validation set while the best performing BERT-tiny model had an accuracy of 50%.

Finally, I also performed additional data cleaning and EDA to get a better understanding of the dataset.
Although this analysis was not used to train any neural nets, this extra EDA did reveal new features of the dataset which may be easy to miss.
For example, I found that shorter reviews had a higher probabilty of corresponding to a rating of 0, that many Goodreads reviewers recieved a free copy of a book in exchange for an honest review, and that a basic sentiment classifier using the NLTK library predicted that, on average, books with a rating of 0 were <em> more positive </em> than books with a rating of 1. The fact reviews with a rating of 0 were surprisingly positive comes from mislabelled/missing data and is explored in the third report.


## <a href="https://api.wandb.ai/links/dmeltzer/s840cljt">Week 2</a>

This week's analysis was mostly the same as Week 1. The exception is the code was refactored so that the processing and training functions are contained in separate python files which can then be imported into a Jupyter notebook.
In addition, I used W&B to perform hyperparameter sweeps for the BERT-tiny model and save the best performing model.
The best performing BERT-tiny model had an accuracy of around 50% on the validation set and only marginally improved on the model studied in the first assignment. That being said, the hyperparameter sweep showed how different choices of learning rate, number of warm-up stes, and number of gradient accumulation steps can effect the accuracy of the model. 

## Report 3

In the final report I performed a new split of the data into train, validation, and test sets. For this new split, in addition to ensuring that each split contained different books, I also stratified by the user_id.
The stratification was useful to avoid imbalanced datasets since some users are more active on Goodreads than others.


