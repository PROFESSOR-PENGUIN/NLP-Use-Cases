import torch
from onnxruntime.quantization import quantize_dynamic,QuantType
import os
import subprocess
import numpy as np
import pandas as pd
import transformers
from pathlib import Path
import streamlit as st

import yaml
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = read_yaml('config.yaml')

zs_chkpt=config['ZEROSHOT_CLF']['zs_chkpt']
zs_mdl_dir=config['ZEROSHOT_CLF']['zs_mdl_dir']
zs_onnx_mdl_dir=config['ZEROSHOT_CLF']['zs_onnx_mdl_dir']
zs_onnx_mdl_name=config['ZEROSHOT_CLF']['zs_onnx_mdl_name']
zs_onnx_quant_mdl_name=config['ZEROSHOT_CLF']['zs_onnx_quant_mdl_name']

zs_mlm_chkpt=config['ZEROSHOT_MLM']['zs_mlm_chkpt']
zs_mlm_mdl_dir=config['ZEROSHOT_MLM']['zs_mlm_mdl_dir']
zs_mlm_onnx_mdl_dir=config['ZEROSHOT_MLM']['zs_mlm_onnx_mdl_dir']
zs_mlm_onnx_mdl_name=config['ZEROSHOT_MLM']['zs_mlm_onnx_mdl_name']

##example
# zero_shot_classification(premise='Tiny worms and breath analyzers could screen for disease while it’s early and treatable',
#                          labels='science, sports, museum')


def zero_shot_classification(premise: str, labels: str, model, tokenizer):
    """

    Args:
        premise:
        labels:
        model:
        tokenizer:

    Returns:

    """
    try:
        labels=labels.split(',')
        labels=[l.lower() for l in labels]
    except:
        raise Exception("please pass atleast 2 labels to classify")

    premise=premise.lower()

    labels_prob=[]

    for l in labels:

        hypothesis= f'this is an example of {l}'

        input = tokenizer.encode(premise,hypothesis,
                             return_tensors='pt',
                                 truncation_strategy='only_first')
        output = model(input)
        entail_contra_prob = output['logits'][:,[0,2]].softmax(dim=1)[:,1].item() #only normalizing entail & contradict probabilties
        labels_prob.append(entail_contra_prob)

    labels_prob_norm=[np.round(100*c/np.sum(labels_prob),1) for c in labels_prob]

    df=pd.DataFrame({'labels':labels,
                     'Probability':labels_prob_norm})

    return df

def create_onnx_model_zs_nli(zs_chkpt,zs_onnx_mdl_dir):
    """

    Args:
        zs_onnx_mdl_dir:

    Returns:

    """

    # create onnx model using
    if not os.path.exists(zs_onnx_mdl_dir):
        try:
            subprocess.run(['python3', '-m', 'transformers.onnx',
                            f'--model={zs_chkpt}',
                            '--feature=sequence-classification',
                            '--atol=1e-3',
                            zs_onnx_mdl_dir])
        except Exception as e:
            print(e)

        # #create quanitzed model from vanila onnx
        # quantize_dynamic(f"{zs_onnx_mdl_dir}/{zs_onnx_mdl_name}",
        #                  f"{zs_onnx_mdl_dir}/{zs_onnx_quant_mdl_name}",
        #                  weight_type=QuantType.QUInt8)
    else:
        pass

def zero_shot_classification_nli_onnx(premise,labels,_session,_tokenizer,hypothesis="This is an example of"):
    """

    Args:
        premise:
        labels:
        _session:
        _tokenizer:
        hypothesis:

    Returns:

    """
    try:
        labels=labels.split(',')
        labels=[l.lower() for l in labels]
    except:
        raise Exception("please pass atleast 2 labels to classify")

    premise=premise.lower()

    labels_prob=[]

    for l in labels:

        hypothesis= f"{hypothesis} {l}"

        inputs = _tokenizer(premise,hypothesis,
                             return_tensors='pt',
                                 truncation_strategy='only_first')

        input_feed = {
            "input_ids": np.array(inputs['input_ids']),
            "attention_mask": np.array((inputs['attention_mask']))
        }

        output = _session.run(output_names=["logits"],input_feed=dict(input_feed))[0] #returns logits as array
        output=torch.from_numpy(output)
        entail_contra_prob = output[:,[0,2]].softmax(dim=1)[:,1].item() #only normalizing entail & contradict probabilties
        labels_prob.append(entail_contra_prob)

    labels_prob_norm=[np.round(100*c/np.sum(labels_prob),1) for c in labels_prob]

    df=pd.DataFrame({'labels':labels,
                     'Probability':labels_prob_norm})

    return df

def create_onnx_model_zs_mlm(zs_mlm_chkpt,zs_mlm_onnx_mdl_dir):
    """

    Args:
        _model:
        _tokenizer:
        zs_mlm_onnx_mdl_dir:

    Returns:

    """
    if not os.path.exists(zs_mlm_onnx_mdl_dir):
        try:
            subprocess.run(['python3', '-m', 'transformers.onnx',
                            f'--model={zs_mlm_chkpt}',
                            '--feature=masked-lm',
                            zs_mlm_onnx_mdl_dir])
        except:
            pass

    else:
        pass

def zero_shot_classification_fillmask_onnx(premise,hypothesis,labels,_session,_tokenizer):
    """

    Args:
        premise:
        hypothesis:
        labels:
        _session:
        _tokenizer:

    Returns:

    """
    try:
        labels=labels.split(',')
        labels=[l.lower().rstrip().lstrip() for l in labels]
    except:
        raise Exception("please pass atleast 2 labels to classify")

    premise=premise.lower()
    hypothesis=hypothesis.lower()

    final_input= f"{premise}.{hypothesis} [MASK]" #this can change depending on chkpt, this is for bert-base-uncased chkpt

    _inputs=_tokenizer(final_input,padding=True, truncation=True,return_tensors="pt")


    ## lowers the performance
    # premise_token_ids=_tokenizer.encode(premise,add_special_tokens=False)
    # hypothesis_token_ids=_tokenizer.encode(hypothesis,add_special_tokens=False)
    #
    # #creating inputs ids
    # input_ids=[_tokenizer.cls_token_id]+premise_token_ids+[_tokenizer.sep_token_id]+hypothesis_token_ids+[_tokenizer.sep_token_id]
    # input_ids=np.array(input_ids)
    #
    # #creating token type ids
    # premise_len=len(premise_token_ids)
    # hypothesis_len=len(hypothesis_token_ids)
    # token_type_ids=np.array([0]*(premise_len+2)+[1]*(hypothesis_len+1))
    #
    # #creating attention mask
    # attention_mask=np.array([1]*(premise_len+hypothesis_len+3))
    #
    # input_feed={
    #     'input_ids': np.expand_dims(input_ids,axis=0),
    #     'token_type_ids': np.expand_dims(token_type_ids,0),
    #     'attention_mask': np.expand_dims(attention_mask,0)
    # }


    input_feed={
        'input_ids': np.array(_inputs['input_ids']),
        'token_type_ids': np.array(_inputs['token_type_ids']),
        'attention_mask': np.array(_inputs['attention_mask'])
    }


    output=_session.run(output_names=['logits'],input_feed=dict(input_feed))[0]

    mask_token_index = np.argwhere(_inputs["input_ids"] == _tokenizer.mask_token_id)[1,0]

    mask_token_logits=output[0,mask_token_index,:]

    #seacrh for logits of input labels
    #encode the labels and get the label id -
    labels_logits=[]
    for l in labels:
        encoded_label=_tokenizer.encode(l)[1]
        labels_logits.append(mask_token_logits[encoded_label])

    #do a softmax on the logits
    labels_logits=np.array(labels_logits)
    labels_logits=torch.from_numpy(labels_logits)
    labels_logits=labels_logits.softmax(dim=0)

    output= {'Labels':labels,
            'Probability':labels_logits}

    df_output = pd.DataFrame(output)
    df_output['Probability'] = df_output['Probability'].apply(lambda x: np.round(100*x, 1))

    return df_output
