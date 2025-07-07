#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from PIL import Image
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from stylecloud import gen_stylecloud
from gensim import corpora
import pyLDAvis.gensim_models
import gensim
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action='ignore')
font_path = 'C:\\Users\\user\\anaconda3\\envs\\text\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\malgun.ttf'


# In[14]:


df1=pd.read_csv('data/텍데분전처리2단계끝.csv')
df=df1.copy()
df=df.iloc[:,1:]


# # Topic 

# ### 1. 정치 및 법(선거, 정부, 법, 검찰, 살인)  약4000개  
# ### 2. 기술(빅데이터, AI, 자율주행, 가상현실, GPT, 로봇)  약 2800개
# ### 3. 경제(투자, 기업, 부자, 주식, 실업, 보험)  약 3000개
# ### 4. 환경(날씨, 재활용)  약 1000개

# #  <font color= red> 1. WordCloud

# ## <font color= green> 1) 정치 및 법 (선거, 정부, 법, 검찰, 살인) 약 4000개
# 

# In[33]:


#슬라이싱의 경우, 토픽별 기사 개수를 활용해 다음 토픽과의 경계 인덱스를 직접 찾아주었습니다
politics= df.iloc[:3848,:].reset_index(drop=True)


# In[35]:


all_data = ''
for _, row in politics.iterrows(): #행의 정보를 담은 객체
    all_data += row['preprocessing_content2']


# In[36]:


all_data


# In[37]:


#https://seaborn.pydata.org/generated/seaborn.color_palette.html
#https://wannabe00.tistory.com/entry/Word-cloud-%EC%9B%90%ED%95%98%EB%8A%94-%EC%83%89%EC%9C%BC%EB%A1%9C-%EA%BE%B8%EB%AF%B8%EA%B8%B0-word-cloud-customize-color
#위의 사이트들을 참고하여 교수님코드와 위의 워드 클라우드 시각화를 추가하여 진행했습니다.


# In[38]:


mask= np.array(Image.open('투명png/politic.png'))

cloud = WordCloud(font_path = font_path,
                  colormap='gist_rainbow',
                  background_color = 'black',
                  collocations=True,
                  width=2000, height=1000,
                 mask=mask)
my_cloud1 = cloud.generate_from_text(all_data)

arr1 = my_cloud1.to_array()

fig = plt.figure(figsize=(10, 10))
plt.imshow(arr1)
plt.axis('off')
plt.show()
fig.savefig('politics.png') #생성한 그림 저장하기


# ## <font color = green> 2) 기술 (빅데이터, AI, 자율주행, 가상현실, GPT) 약 2800개

# In[39]:


tech= df.iloc[3848:6458,:].reset_index(drop=True)


# In[40]:


all_data = ''
for _, row in tech.iterrows(): #행의 정보를 담은 객체
    all_data += row['preprocessing_content2']


# In[41]:


mask= np.array(Image.open('투명png/ai.png'))

cloud = WordCloud(font_path = font_path,
                  colormap='gist_rainbow',
                  background_color = 'black',
                  collocations=True,
                  width=5000, height=4000,
                 mask=mask)
my_cloud1 = cloud.generate_from_text(all_data)

arr1 = my_cloud1.to_array()

fig = plt.figure(figsize=(10, 10))
plt.imshow(arr1)
plt.axis('off')
plt.show()
fig.savefig('tech.png') #생성한 그림 저장하기


# ## <font color  =green> 3) 경제 (투자, 기업, 부자, 주식, 실업, 보험) 약 3000개

# In[42]:


money=df.iloc[6458:9375,:].reset_index(drop=True)


# In[43]:


all_data = ''
for _, row in money.iterrows(): #행의 정보를 담은 객체
    all_data += row['preprocessing_content2']


# In[44]:


mask= np.array(Image.open('투명png/won.png'))

cloud = WordCloud(font_path = font_path,
                  colormap='gist_rainbow',
                  background_color = 'black',
                  collocations=True,
                  width=5000, height=4000,
                 mask=mask)
my_cloud1 = cloud.generate_from_text(all_data)

arr1 = my_cloud1.to_array()

fig = plt.figure(figsize=(10, 10))
plt.imshow(arr1)
plt.axis('off')
plt.show()
fig.savefig('money.png') #생성한 그림 저장하기


# ## <font color = green> 4) 환경 (날씨, 재활용)  약 1000개

# In[45]:


env=df.iloc[9375:,:].reset_index(drop=True)


# In[46]:


all_data = ''
for _, row in env.iterrows(): #행의 정보를 담은 객체
    all_data += row['preprocessing_content2']


# In[47]:


mask= np.array(Image.open('투명png/env.png'))

cloud = WordCloud(font_path = font_path,
                  colormap='gist_rainbow',
                  background_color = 'black',
                  collocations=True,
                  width=5000, height=4000,
                 mask=mask)
my_cloud1 = cloud.generate_from_text(all_data)

arr1 = my_cloud1.to_array()

fig = plt.figure(figsize=(10, 10))
plt.imshow(arr1)
plt.axis('off')
plt.show()
fig.savefig('env.png') #생성한 그림 저장하기


# # <font color= red> 2. Topic Modeling

# In[15]:


# LDA를 위해서는 리스트 형태로 바꿔주어야 한다
df['preprocessing_content2'] = df['preprocessing_content2'].apply(lambda x: x.split())


# In[16]:


word_dict = corpora.Dictionary(df['preprocessing_content2'])


# In[17]:


corpus = [word_dict.doc2bow(text) for text in df['preprocessing_content2']]


# In[18]:


len(word_dict)


# ## <font color = red> 2-1) topic 수: 4  -> 처음에 세웠던 토픽 4개(정치, 기술, 경제, 환경)로 토픽 모델링이 잘 될까? 를 확인해봄

# In[19]:


N_TOPICS = 4
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = N_TOPICS, id2word=word_dict, passes = 15)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# In[20]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, word_dict)
pyLDAvis.display(vis)


# ## <font color= red>  2-2) topic 수 : 19  -> 세분화한 topic으로 더 잘 나눌 수 있는지 보기 위해 19개로 선정

# In[21]:


N_TOPICS = 19
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = N_TOPICS, id2word=word_dict, passes = 15)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# In[22]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, word_dict)
pyLDAvis.display(vis)

