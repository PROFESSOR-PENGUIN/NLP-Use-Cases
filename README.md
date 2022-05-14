Summary of Tasks 
=======================
| Tasks                          | Models                                          | Runtime        |
|--------------------------------|-------------------------------------------------|----------------|
| Sentiment Classification       | distilbert-base-uncased-finetuned-sst-2-english | ONNX           |
| Zero Shot Classification NLI   | distilbart-mnli-12-1                            | ONNX           |
| Zero Shot Classification MLM   | bert-base-uncased                               | ONNX           |

Running the APP
=======================
1. Clone the Repo
2. Change the directory to NLP-Use-Cases
   1. cd NLP-Use-Cases
3. Activate the virtual Environment
   1. On Mac -  source &nbsp; /venv/bin/activate
   2. On Windows - venv\Scripts\activate
4. Install the requirements
   1. pip install -r requirements.txt
5. Run the App
   1. streamlit run app.py


