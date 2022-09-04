from huggingface_hub import notebook_login
from huggingface_hub import HfFolder


#enter your API key, you can make one for free on HF
notebook_login()

from huggingface_hub import InferenceApi

inference = InferenceApi("bigscience/bloom",token="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")

from IPython.display import HTML as html_print

def cstr(s, color='black'):
    #return "<text style=color:{}>{}</text>".format(color, s)
    return "<text style=color:{}>{}</text>".format(color, s.replace('\n', '<br>'))

def cstr_with_newlines(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s.replace('\n', '<br>'))


def infer(prompt,
          max_length = 128,
          top_k = 0,
          num_beams = 0,
          no_repeat_ngram_size = 2,
          top_p = 0.9,
          seed=42,
          temperature=0.7,
          greedy_decoding = False,
          return_full_text = False):
    

    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams,
        "return_full_text":return_full_text
    }
    
    response = inference(prompt, params=params)
    return html_print(cstr(prompt, color='#f1f1c7') + cstr(response[0]['generated_text'], color='#a1d8eb')), response[0]['generated_text']

import streamlit as st

st.title("Chat with BLOOM ^^")
user_input = st.text_area("Enter a question, a sentence, a complex input...")


if user_input:
    color_resp, resp = infer(user_input)
    st.write(color_resp)
else:
    pass
