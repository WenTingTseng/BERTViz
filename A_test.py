#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"')
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
get_ipython().system('test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo')
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']
get_ipython().system('pip install regex')


# In[2]:


get_ipython().system('pip3 install transformers')


# In[3]:


from bertviz import head_view
from transformers import BertTokenizer, BertForSequenceClassification


# In[4]:


from bertviz import head_view
from transformers import BertTokenizer, BertModel
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW


# In[5]:


get_ipython().run_cell_magic('javascript', '', "require.config({\n  paths: {\n      d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',\n      jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',\n  }\n});")


# In[6]:


def show_head_view(model, tokenizer, sentence_a, sentence_b=None):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if sentence_b:     
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids)[-1]
        sentence_b_start = None
      
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

    head_view(attention, tokens, sentence_b_start)


# In[17]:


from IPython.display import clear_output
# 在 jupyter notebook 裡頭顯示 visualzation 的 helper
def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

clear_output()


# In[19]:


tokenizer = BertTokenizer(vocab_file='C:\\Users\\Acer\\bertviz-master\\bertviz-master\\model\\bert-base-chinese-vocab.txt')
bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
config = bert_config.from_pretrained('C:\\Users\\Acer\\bertviz-master\\bertviz-master\\model3\\config.json',output_attentions=True)
model = bert_class.from_pretrained('C:\\Users\\Acer\\bertviz-master\\bertviz-master\\model3\\pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)


# In[21]:


# 情境 1 的句子

# 情境 1 的句子
sentence_a = "我要怎麼做才能使用YouBike微笑單車服務？"
sentence_b = "我要怎麼做才能使用YouBike微笑單車服務？"


# 得到 tokens 後丟入 BERT 取得 attention
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = model(input_ids, token_type_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html()

# # 交給 BertViz 視覺化
# head_view(attention, tokens)


# In[22]:


head_view(attention, tokens)


# In[ ]:




