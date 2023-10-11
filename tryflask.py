from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
print("Current Working Directory:", os.getcwd())

app = Flask(__name__)

# Load the BERT model and data outside the route function to avoid loading them on every request
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.to(device)

data = pd.read_csv("C:/Users/justi/Desktop/Flask/Resume.csv")
resumes = data['Resume_str'].tolist()
categories = data['Category'].tolist()
data = data.drop(columns=['Resume_html'])

# Load other necessary functions and variables
noise_words = ['n a', 'company name', 'city', 'state', r'\[YEAR\]', r'\[NUMBER\]'] 

def remove_noise_words(text, noise_words):
    for word in noise_words:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.get_json()["job_description"]
        job_desc_tokens = tokenizer(job_description, padding=True, truncation=True, return_tensors='pt')
        job_desc_tokens = job_desc_tokens.to(device)
        
        resume_embeddings = []
        for index, (resume, category) in enumerate(zip(resumes, categories)):
            if isinstance(resume, str):
                cleaned_resume = remove_noise_words(resume, noise_words)
                tokenized_resume = tokenizer(cleaned_resume, padding='max_length', truncation=True, return_tensors='pt')
                tokenized_resume = tokenized_resume.to(device)
                
                with torch.no_grad():
                    resume_emb = model(**tokenized_resume).last_hidden_state[:, 0, :]
                    resume_emb = resume_emb.cpu()
                resume_embeddings.append(resume_emb)
            else:
                continue
            
            # Limit the loop to process only the first five resumes FOR CONCEPT!!
            if index >= 9:
                break
            
        with torch.no_grad():
            job_desc_embeddings = model(**job_desc_tokens).last_hidden_state[:, 0, :]
            job_desc_embeddings = job_desc_embeddings.cpu()
        
        similarities = []
        for resume_emb, category in zip(resume_embeddings, categories):
            similarity = cosine_similarity(job_desc_embeddings, resume_emb)
            similarities.append(similarity.item())
        similarities = np.array(similarities)
        
        least_similar_index = np.argmin(similarities)
        least_similar_category = categories[least_similar_index]
        least_similar_similarity = similarities[least_similar_index]
        least_similar_resume = resumes[least_similar_index]
        
        top_similar_indices = similarities.argsort()[-10:][::-1]
        top_similar_resumes = [(categories[index], resumes[index][:200]) for index in top_similar_indices]
        
        return jsonify({
            "least_similar_category": least_similar_category,
            "least_similar_resume": least_similar_resume[:200],
            "least_similar_similarity": least_similar_similarity,
            "top_similar_resumes": top_similar_resumes
        })
    
    return render_template('ResumeChecker.html')

if __name__ == '__main__':
    app.run(debug=True)
