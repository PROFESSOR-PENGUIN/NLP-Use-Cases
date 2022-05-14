import numpy as np
import transformers
from onnxruntime.quantization import quantize_dynamic,QuantType
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path
import os
import torch
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = read_yaml('config.yaml')

sent_chkpt=config['SENTIMENT_CLF']['sent_chkpt']
sent_mdl_dir=config['SENTIMENT_CLF']['sent_mdl_dir']
sent_onnx_mdl_dir=config['SENTIMENT_CLF']['sent_onnx_mdl_dir']
sent_onnx_mdl_name=config['SENTIMENT_CLF']['sent_onnx_mdl_name']
sent_onnx_quant_mdl_name=config['SENTIMENT_CLF']['sent_onnx_quant_mdl_name']

def classify_sentiment(texts,model,tokenizer):
    """
        user will pass texts separated by comma
    """
    try:
        texts=texts.split(',')
    except:
        pass

    input = tokenizer(texts, padding=True, truncation=True,
                      return_tensors="pt")
    logits = model(**input)['logits'].softmax(dim=1)
    logits = torch.argmax(logits, dim=1)
    output = ['Positive' if i == 1 else 'Negative' for i in logits]
    return output


def create_onnx_model_sentiment(_model, _tokenizer,sent_onnx_mdl_dir=sent_onnx_mdl_dir):
    """

    Args:
        _model: model checkpoint with AutoModelForSequenceClassification
        _tokenizer: model checkpoint with AutoTokenizer

    Returns:
        Creates a simple ONNX model & int8 Quantized Model in the directory "sent_clf_onnx/" if directory not present

    """
    if not os.path.exists(sent_onnx_mdl_dir):
        try:
            os.mkdir(sent_onnx_mdl_dir)
        except:
            pass
        pipeline=transformers.pipeline("text-classification", model=_model, tokenizer=_tokenizer)

        onnx_convert.convert_pytorch(pipeline,
                                     opset=11,
                                     output=Path(f"{sent_onnx_mdl_dir}/{sent_onnx_mdl_name}"),
                                     use_external_format=False
                                     )

        # quantize_dynamic(f"{sent_onnx_mdl_dir}/{sent_onnx_mdl_name}",
        #                  f"{sent_onnx_mdl_dir}/{sent_onnx_quant_mdl_name}",
        #                  weight_type=QuantType.QUInt8)
    else:
        pass


def classify_sentiment_onnx(texts, _session, _tokenizer):
    """

    Args:
        texts: input texts from user
        _session: pass ONNX runtime session
        _tokenizer: Relevant Tokenizer e.g. AutoTokenizer.from_pretrained("same checkpoint as the model")

    Returns:
        list of Positve and Negative texts

    """
    try:
        texts=texts.split(',')
    except:
        pass

    _inputs = _tokenizer(texts, padding=True, truncation=True,
                      return_tensors="np")

    input_feed={
        "input_ids":np.array(_inputs['input_ids']),
        "attention_mask":np.array((_inputs['attention_mask']))
    }

    output = _session.run(input_feed=input_feed, output_names=['output_0'])[0]

    output=np.argmax(output,axis=1)
    output = ['Positive' if i == 1 else 'Negative' for i in output]
    return output

