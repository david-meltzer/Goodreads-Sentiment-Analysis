import numpy as np
import requests
import streamlit as st
import json
def main():


    
    # Use feature-extraction API to get sentence embeddings.
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    # Token to access Huggingface Inference API.
    headers = {"Authorization": f"Bearer {st.secrets['HF_token']}"}


    st.title("Semantic Search for Questions on Reddit.")

    st.write("This application lets you perform sentiment analysis on book reviews.\
        Simply input a review into the text below and the application will give two predictions for what the \
             rating is on a scale of 0-5. The models will also produce the score they assigned their prediction. The score is\
             between 0 and 1 and quantifies the confidence the model has in its prediction.\
             \n\n \
             Specifically, we consider two pre-trained models, [BERT-tiny](https://huggingface.co/dhmeltzer/bert-tiny-goodreads-wandb) and [DistilBERT](https://huggingface.co/dhmeltzer/distilbert-goodreads-wandb)\
              which have been fine-tuned on a dataset of Goodreads book \
             reviews, see [here](https://www.kaggle.com/competitions/goodreads-books-reviews-290312/data) for the original dataset. \
             These models are deployed on AWS and are accessed using a REST API. To deploy the models we used a combination of AWS Sagemaker, Lambda, and API Gateway.\
             \n\n \
             To read more about this project and specifically how we cleaned the data and trained the models, see the following GitHub (repository)[https://github.com/david-meltzer/Goodreads-Sentiment-Analysis].")


    AWS_key = st.secrets['AWS-key']

    checkpoints = {}
    checkpoints['DistilBERT'] = 'https://85a720iwy2.execute-api.us-east-1.amazonaws.com/add_apis/distilbert-goodreads'
    checkpoints['BERT-tiny'] = 'https://055dugvmzl.execute-api.us-east-1.amazonaws.com/beta/'
    
    # User search with default question.
    user_input = st.text_area("Search box", "I loved the Lord of the Rings trilogy. \
                               It is a classic and beautifully written story and J.R.R. Tolkein really made Middle-Earth come to life. \
                              My favorite part of the book though was when the hobbits met Tom Bombadil, it's too bad he was not in the movies.")


    convert_dict = {}
    for i in range(6):
        convert_dict[f'LABEL_{i}'] = i

    # Fetch results
    if user_input:
        # Get IDs for each search result. 
        for model_name, URL in checkpoints.items():

            headers={'x-api-key': AWS_key}

            input_data = json.dumps({'inputs':user_input})
            r = requests.post(URL,
                  data=input_data,
                  headers=headers).json()[0]
 
            label, score = convert_dict[r['label']], r['score']

            st.write(f"**Model Name**: {model_name}")
            st.write(f"**Predicted Review**: {label}")
            st.write(f"**Confidence**: {score}")
            st.write("-"*20) 

if __name__ == "__main__":
    main()
