import numpy as np
import pandas as pd
import streamlit as st
from streamlit_text_rating.st_text_rater import st_text_rater
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
import onnxruntime as ort
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
global _plotly_config
_plotly_config={'displayModeBar': False}

from sentiment_clf_helper import (classify_sentiment,
                                  create_onnx_model_sentiment,
                                  classify_sentiment_onnx)

from zeroshot_clf_helper import (zero_shot_classification,
                                 create_onnx_model_zs_nli,
                                 create_onnx_model_zs_mlm,
                                 zero_shot_classification_nli_onnx,
                                 zero_shot_classification_fillmask_onnx)

import multiprocessing
total_threads=multiprocessing.cpu_count()#for ort inference

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

zs_chkpt=config['ZEROSHOT_CLF']['zs_chkpt']
zs_mdl_dir=config['ZEROSHOT_CLF']['zs_mdl_dir']
zs_onnx_mdl_dir=config['ZEROSHOT_CLF']['zs_onnx_mdl_dir']
zs_onnx_mdl_name=config['ZEROSHOT_CLF']['zs_onnx_mdl_name']
zs_onnx_quant_mdl_name=config['ZEROSHOT_CLF']['zs_onnx_quant_mdl_name']

zs_mlm_chkpt=config['ZEROSHOT_MLM']['zs_mlm_chkpt']
zs_mlm_mdl_dir=config['ZEROSHOT_MLM']['zs_mlm_mdl_dir']
zs_mlm_onnx_mdl_dir=config['ZEROSHOT_MLM']['zs_mlm_onnx_mdl_dir']
zs_mlm_onnx_mdl_name=config['ZEROSHOT_MLM']['zs_mlm_onnx_mdl_name']

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='None',  # String or None. Strings get appended with "• Streamlit".
)

padding_top = 0
st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

def set_page_title(title):
    st.sidebar.markdown(unsafe_allow_html=True, body=f"""
        <iframe height=0 srcdoc="<script>
            const title = window.parent.document.querySelector('title') \

            const oldObserver = window.parent.titleObserver
            if (oldObserver) {{
                oldObserver.disconnect()
            }} \

            const newObserver = new MutationObserver(function(mutations) {{
                const target = mutations[0].target
                if (target.text !== '{title}') {{
                    target.text = '{title}'
                }}
            }}) \

            newObserver.observe(title, {{ childList: true }})
            window.parent.titleObserver = newObserver \

            title.text = '{title}'
        </script>" />
    """)


set_page_title('NLP use cases')

# Hide Menu Option
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#onnx runtime inference threading changes -- session options must be passed in session run
# os.environ["OMP_NUM_THREADS"] = "1" #use this before changing session options of onnx runtime
session_options_ort = ort.SessionOptions()
session_options_ort.intra_op_num_threads=1
session_options_ort.inter_op_num_threads=1
# session_options_ort.execution_mode = session_options_ort.ExecutionMode.ORT_SEQUENTIAL

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=None, ttl=None)
def create_model_dir(chkpt, model_dir,task_type):
    if not os.path.exists(model_dir):
        try:
            os.mkdir(path=model_dir)
        except:
            pass
        if task_type=='classification':
            _model = AutoModelForSequenceClassification.from_pretrained(chkpt)
            _tokenizer = AutoTokenizer.from_pretrained(chkpt)
            _model.save_pretrained(model_dir)
            _tokenizer.save_pretrained(model_dir)
        elif task_type=='mlm':
            _model=AutoModelForMaskedLM.from_pretrained(chkpt)
            _tokenizer=AutoTokenizer.from_pretrained(chkpt)
            _model.save_pretrained(model_dir)
            _tokenizer.save_pretrained(model_dir)
        else:
            pass
    else:
        pass


#title using markdown
st.markdown("<h1 style='text-align: center; color: #3366ff;'>NLP Basic Use Cases</h1>", unsafe_allow_html=True)
st.markdown("---")
with st.sidebar:
    # title using markdown
    st.markdown("<h1 style='text-align: left; color: ;'>NLP Tasks</h1>", unsafe_allow_html=True)
    select_task=st.selectbox(label="Select task from drop down menu",
                 options=['README',
                          'Detect Sentiment','Zero Shot Classification'])

############### Pre-Download & instantiate objects for sentiment analysis *********************** START **********************

# #create model/token dir for sentiment classification for faster inference
create_model_dir(chkpt=sent_chkpt, model_dir=sent_mdl_dir,task_type='classification')


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=None, ttl=None)
def sentiment_task_selected(task,
                            sent_chkpt=sent_chkpt,
                            sent_mdl_dir=sent_mdl_dir,
                            sent_onnx_mdl_dir=sent_onnx_mdl_dir,
                            sent_onnx_mdl_name=sent_onnx_mdl_name,
                            sent_onnx_quant_mdl_name=sent_onnx_quant_mdl_name):
    ##model & tokenizer initialization for normal sentiment classification
    # model_sentiment=AutoModelForSequenceClassification.from_pretrained(sent_chkpt)
    # tokenizer_sentiment=AutoTokenizer.from_pretrained(sent_chkpt)
    tokenizer_sentiment = AutoTokenizer.from_pretrained(sent_mdl_dir)

    # # create onnx model for sentiment classification but once created in your local app comment this out
    # create_onnx_model_sentiment(_model=model_sentiment, _tokenizer=tokenizer_sentiment)

    #create inference session
    sentiment_session = ort.InferenceSession(f"{sent_onnx_mdl_dir}/{sent_onnx_mdl_name}",sess_options=session_options_ort)
    # sentiment_session_quant = ort.InferenceSession(f"{sent_onnx_mdl_dir}/{sent_onnx_quant_mdl_name}")

    return tokenizer_sentiment,sentiment_session

############## Pre-Download & instantiate objects for sentiment analysis ********************* END **********************************


############### Pre-Download & instantiate objects for Zero shot clf NLI *********************** START **********************

# create model/token dir for zeroshot clf -- already created so not required
create_model_dir(chkpt=zs_chkpt, model_dir=zs_mdl_dir,task_type='classification')

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=None, ttl=None)
def zs_nli_task_selected(task,
                     zs_chkpt ,
                     zs_mdl_dir,
                     zs_onnx_mdl_dir,
                     zs_onnx_mdl_name):

    ##model & tokenizer initialization for normal ZS classification
    # model_zs=AutoModelForSequenceClassification.from_pretrained(zs_chkpt)
    # we just need tokenizer for inference and not model since onnx model is already saved
    # tokenizer_zs=AutoTokenizer.from_pretrained(zs_chkpt)
    tokenizer_zs = AutoTokenizer.from_pretrained(zs_mdl_dir)

    ## create onnx model for zeroshot but once created locally comment it out.
    #create_onnx_model_zs_nli()

    #create inference session from onnx model
    zs_session = ort.InferenceSession(f"{zs_onnx_mdl_dir}/{zs_onnx_mdl_name}",sess_options=session_options_ort)

    return tokenizer_zs,zs_session

############## Pre-Download & instantiate objects for Zero shot NLI analysis ********************* END **********************************


############### Pre-Download & instantiate objects for Zero shot clf NLI *********************** START **********************
## create model/token dir for zeroshot clf -- already created so not required
# create_model_dir(chkpt=zs_mlm_chkpt, model_dir=zs_mlm_mdl_dir, task_type='mlm')

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=None, ttl=None)
def zs_mlm_task_selected(task,
                         zs_mlm_chkpt=zs_mlm_chkpt,
                         zs_mlm_mdl_dir=zs_mlm_mdl_dir,
                         zs_mlm_onnx_mdl_dir=zs_mlm_onnx_mdl_dir,
                         zs_mlm_onnx_mdl_name=zs_mlm_onnx_mdl_name):
    ##model & tokenizer initialization for normal ZS classification
    # model_zs_mlm=AutoModelForMaskedLM.from_pretrained(zs_mlm_mdl_dir)
    ##we just need tokenizer for inference and not model since onnx model is already saved
    # tokenizer_zs_mlm=AutoTokenizer.from_pretrained(zs_mlm_chkpt)
    tokenizer_zs_mlm = AutoTokenizer.from_pretrained(zs_mlm_mdl_dir)

    # # create onnx model for zeroshot but once created locally comment it out.
    # create_onnx_model_zs_mlm(_model=model_zs_mlm,
    #                          _tokenizer=tokenizer_zs_mlm,
    #                          zs_mlm_onnx_mdl_dir=zs_mlm_onnx_mdl_dir)

    # create inference session from onnx model
    zs_session_mlm = ort.InferenceSession(f"{zs_mlm_onnx_mdl_dir}/{zs_mlm_onnx_mdl_name}", sess_options=session_options_ort)

    return tokenizer_zs_mlm, zs_session_mlm


############## Pre-Download & instantiate objects for Zero shot MLM analysis ********************* END **********************************

# Image.open('hf_space1.png').convert('RGB').save('hf_space1.png')
img = Image.open("hf_space1.png")

if select_task=='README':
    st.header("NLP Summary")
    st.write(f"The App gives you ability to 1) Detect Sentiment, 2) Zeroshot Classification.Currently.It has {total_threads} CPU cores but only 1 is available per user so "
             f"inference time will be on the higher side.")
    st.markdown("---")
    st.image(img)

if select_task == 'Detect Sentiment':
    t1=time.time()
    tokenizer_sentiment,sentiment_session = sentiment_task_selected(task=select_task)
    ##below 2 steps are slower as caching is not enabled
    # tokenizer_sentiment = AutoTokenizer.from_pretrained(sent_mdl_dir)
    # sentiment_session = ort.InferenceSession(f"{sent_onnx_mdl_dir}/{sent_onnx_mdl_name}")
    t2 = time.time()
    st.write(f"Total time to load Model is {(t2-t1)*1000:.1f} ms")

    st.subheader("You are now performing Sentiment Analysis")
    input_texts = st.text_input(label="Input texts separated by comma")
    c1,c2,_,_=st.columns(4)

    with c1:
        response1=st.button("Compute (ONNX runtime)")

    if response1:
        start = time.time()
        sentiments=classify_sentiment_onnx(input_texts,
                                           _session=sentiment_session,
                                           _tokenizer=tokenizer_sentiment)
        end = time.time()
        st.write(f"Time taken for computation {(end - start) * 1000:.1f} ms")

        for i,t in enumerate(input_texts.split(',')):
            if sentiments[i]=='Positive':
                response=st_text_rater(t + f"--> This statement is {sentiments[i]}",
                                       color_background='rgb(154,205,50)',key=t)
            else:
                response = st_text_rater(t + f"--> This statement is {sentiments[i]}",
                                         color_background='rgb(233, 116, 81)',key=t)
    else:
        pass

if select_task=='Zero Shot Classification':
    t1=time.time()
    tokenizer_zs,session_zs = zs_nli_task_selected(task=select_task ,
                                                   zs_chkpt=zs_chkpt,
                                                   zs_mdl_dir=zs_mdl_dir,
                                                   zs_onnx_mdl_dir=zs_onnx_mdl_dir,
                                                   zs_onnx_mdl_name=zs_onnx_mdl_name)
    t2 = time.time()
    st.write(f"Total time to load NLI Model is {(t2-t1)*1000:.1f} ms")

    t1=time.time()
    tokenizer_zs_mlm,session_zs_mlm = zs_mlm_task_selected(task=select_task,
                                                           zs_mlm_chkpt=zs_mlm_chkpt,
                                                           zs_mlm_mdl_dir=zs_mlm_mdl_dir,
                                                           zs_mlm_onnx_mdl_dir=zs_mlm_onnx_mdl_dir,
                                                           zs_mlm_onnx_mdl_name=zs_mlm_onnx_mdl_name)
    t2 = time.time()
    st.write(f"Total time to load MLM Model is {(t2-t1)*1000:.1f} ms")

    st.subheader("Zero Shot Classification using NLI & MLM")
    input_texts = st.text_input(label="Input text to classify into topics")
    input_lables = st.text_input(label="Enter labels separated by commas")
    input_hypothesis = st.text_input(label="Enter your hypothesis",value="This is an example of")

    c1,c2,_,=st.columns(3)

    with c1:
        response1=st.button("Compute using NLI approach (ONNX runtime)")

    with c2:
        response2=st.button("Compute using Fill-Mask approach(ONNX runtime)")

    if response1:
        start = time.time()
        df_output = zero_shot_classification_nli_onnx(premise=input_texts,
                                                      labels=input_lables,
                                                      hypothesis=input_hypothesis,
                                                      _session=session_zs,
                                                     _tokenizer=tokenizer_zs,
                                                      )
        end = time.time()
        st.write(f"Time taken for computation {(end-start)*1000:.1f} ms")
        fig = px.bar(x='Probability',
                     y='labels',
                     text='Probability',
                     data_frame=df_output,
                     title='Zero Shot NLI Normalized Probabilities')

        st.plotly_chart(fig, config=_plotly_config)

    elif response2:
        start=time.time()
        df_output=zero_shot_classification_fillmask_onnx(premise=input_texts,
                                                         labels=input_lables,
                                                         hypothesis=input_hypothesis,
                                                         _session=session_zs_mlm,
                                                        _tokenizer=tokenizer_zs_mlm,
                                                         )
        end=time.time()
        st.write(f"Time taken for computation {(end - start) * 1000:.1f} ms")
        st.write(f"Currently hypothesis and premise have *single token_type_ids*  ."
                 f"Once updated for different *token_type_ids* expect the model performance to increase.")

        fig = px.bar(x='Probability',
                     y='Labels',
                     text='Probability',
                     data_frame=df_output,
                     title='Zero Shot MLM Normalized Probabilities')

        st.plotly_chart(fig, config=_plotly_config)
    else:
        pass


