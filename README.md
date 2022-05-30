---
title: Sentiment & Zeroshot Classification using Streamlit
app_file: app.py
---

Summary of Tasks 
=======================
| Tasks                          | Models                                          | Runtime        |
|--------------------------------|-------------------------------------------------|----------------|
| Sentiment Classification       | distilbert-base-uncased-finetuned-sst-2-english | ONNX           |
| Zero Shot Classification NLI   | distilbart-mnli-12-1                            | ONNX           |
| Zero Shot Classification MLM   | bert-base-uncased                               | ONNX           |

Running the APP
=======================
1. Clone the Repo (make sure you have git lfs installed for large files)
2. Change the directory to NLP-Use-Cases
   1. cd NLP-Use-Cases
3. Create Virtual env
   1. python3 -m venv venv
4. Activate the virtual Environment
   1. On Mac -  source &nbsp; /venv/bin/activate
   2. On Windows - venv\Scripts\activate
5. Make Sure your pip is updated using
   1. python3 -m pip install --upgrade pip
6. Install the requirements
   1. pip3 install -r requirements.txt
7. Run the App
   1. streamlit run app.py


