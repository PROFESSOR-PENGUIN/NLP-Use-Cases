from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np
import subprocess
import os


#create onnx model using
if not os.path.exists("zs_model_onnx"):
    try:
        subprocess.run(['python3','-m','transformers.onnx',
                        '--model=facebook/bart-large-mnli',
                        '--feature=sequence-classification',
                        'zs_model_onnx/'])
    except:
        pass

#create session of saved onnx model
session = InferenceSession("zs_model_onnx/model.onnx")

#tokenizer for the chkpt
tokenizer=AutoTokenizer.from_pretrained('zs_model_dir')

# ONNX Runtime expects NumPy arrays as input
inputs = tokenizer("Using DistilBERT with ONNX Runtime!","you know how", return_tensors="np")
input_feed = {
    "input_ids": np.array(inputs['input_ids']),
    "attention_mask": np.array((inputs['attention_mask']))
}

#output
outputs = session.run(output_names=["logits"], input_feed=dict(input_feed))

print(outputs)
