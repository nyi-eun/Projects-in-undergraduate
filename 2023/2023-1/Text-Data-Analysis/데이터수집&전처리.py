#!/usr/bin/env python
# coding: utf-8

# In[226]:


from bs4 import BeautifulSoup
import urllib
from selenium import webdriver
import os
import sys
import urllib.request
import json
import re
import time
import pandas as pd
import nltk
from tqdm import tqdm
from konlpy.tag import Okt
from nltk.tokenize import word_tokenize

#driver = webdriver.Chrome('D:\MyWorkspace\chromedriver.exe')


# # 1. 데이터 수집

# ## <font color= 'red'> 0) BASE CODE <함수 정의, 네이버 뉴스 본문 크롤링>

# In[8]:


def remove_tag(my_str):
    ## 태그를 지우는 함수
    p = re.compile('(<([^>]+)>)')
    return p.sub('', my_str)

def sub_html_special_char(my_str):
    ## 특수문자를 나타내는 &apos;, &quot를 실제 특수문자로 변환
    p1 = re.compile('&lt;') #lt를 <로 바꿔줘
    p2 = re.compile('&gt;')
    p3 = re.compile('&amp;')
    p4 = re.compile('&apos;')
    p5 = re.compile('&quot;')

    result = p1.sub('\<', my_str)
    result = p2.sub('\>', result)
    result = p3.sub('\&', result)
    result = p4.sub('\'', result)
    result = p5.sub('\"', result)
    return result


# In[9]:


base_url = 'https://openapi.naver.com/v1/search/news.json'

def getresult(client_id,client_secret,query,n_display,start,sort='sim'):
    encQuery = urllib.parse.quote(query)
    
    url = f'{base_url}?query={encQuery}&display={n_display}&start={start}&sort={sort}'
    my_request = urllib.request.Request(url)
    my_request.add_header("X-Naver-Client-Id",client_id)
    my_request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(my_request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        search_result_str = response_body.decode('utf-8')
        search_results = json.loads(search_result_str)
    else:
        print("Error Code:" + rescode)
    return search_results['items']


# In[10]:


#데이터 수집 과정 중 일부는 배민성 학우분과 같이 협업하였습니다.
#slack에 이수인 학우분이 올려주신 코드도 참고하였습니다.
def search_news_with_link (result):
    article_ids = ['dic_area']
    titles = []
    links = []
    pubdates = []
    contents = []

    p = re.compile('https://n.news.naver.com/.+')
    for i, item in enumerate(result):
        if p.match(item['link']): ## <link>태그의 문자열이 n.news.naver.com/으로 시작하는 결과만 추출
            title = sub_html_special_char(remove_tag(item['title']))
            link = item['link']
            pubdate = item['pubDate']
            titles.append(title)
            links.append(link)
            pubdates.append(pubdate)

            html = urllib.request.urlopen(link)
            bs_obj = BeautifulSoup(html, 'html.parser')
            for article_id in article_ids:
                print(article_id)
                content = bs_obj.find_all('div', {'id':article_id})
                if len(content) > 0:
                    contents.append(content[0].text)
                    break
                else:
                    contents.append(0)
                    #연예뉴스와 같은 뉴스들을 수집하지 않기 위해 위와 같은 코드를 작성함
                    #dic_area가 아닌 본문 id들은 0으로 채워준 후, 모든 데이터 수합 후 삭제함
                    #데이터 프레임에 추가할 때, 0과 같은 값을 채워주지 않을 경우, 데이터 프레임이 만들어지지 않음(길이 문제 발생)
                
    result_dict = {'title': titles, 'link': links, 'pubdate': pubdates, 'content': contents}
    df = pd.DataFrame.from_dict(result_dict)
    return df


# In[11]:


client_id = '03xivz4Z_LC7mSS5tkO6'
client_secret = 'uvknoLe1Jq'
n_display=100


# # <font color= 'red'> 1) data crawling - 4 topic 

# ### 1. 정치 및 법(선거, 정부, 법, 검찰, 살인)  
# ### 2. 기술(빅데이터, AI, 자율주행, 가상현실, GPT, 로봇)  
# ### 3. 경제(투자, 기업, 부자, 주식, 실업, 보험)  
# ### 4. 환경(날씨, 재활용)  

# # 1. 정치 및 법 (선거, 정부, 법, 검찰, 살인)

# # 대선

# In[22]:


query = '대선'


# In[23]:


#일부 코드는 배민성 학우분과 협업하였습니다.
#slack에 올려주신 이수인 학우분의 코드도 일부 참고하였습니다.
total_results = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results = pd.concat([total_results, up_result])
    print(len(total_results))


# In[24]:


total_results.reset_index(drop=True, inplace=True)


# # 정부

# In[15]:


query = '정부'


# In[16]:


total_results2 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results2 = pd.concat([total_results2
                                , up_result])
    print(len(total_results2))


# In[18]:


total_results2.reset_index(drop=True, inplace=True)


# # 법

# In[25]:


query = '법'


# In[26]:


total_results3 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results3 = pd.concat([total_results3, up_result])
    print(len(total_results3))


# In[27]:


total_results3.reset_index(drop=True, inplace=True)


# # 검찰

# In[30]:


query = '검찰'


# In[32]:


total_results4 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results4 = pd.concat([total_results4, up_result])
    print(len(total_results4))


# In[33]:


total_results4.reset_index(drop=True, inplace=True)


# # 살인

# In[35]:


query = '살인'


# In[36]:


total_results5 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results5 = pd.concat([total_results5, up_result])
    print(len(total_results5))


# In[37]:


total_results5.reset_index(drop=True, inplace=True)


# # 2. 기술 (빅데이터, AI, 자율주행, 가상현실, gpt, 로봇)

# # 빅데이터

# In[39]:


query = '빅데이터'


# In[40]:


total_results6 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results6 = pd.concat([total_results6, up_result])
    print(len(total_results6))


# In[41]:


total_results6.reset_index(drop=True, inplace=True)


# # AI

# In[43]:


query = 'AI'


# In[44]:


total_results7 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results7 = pd.concat([total_results7, up_result])
    print(len(total_results7))


# In[45]:


total_results7.reset_index(drop=True, inplace=True)


# # 자율주행

# In[46]:


query = '자율주행'


# In[48]:


total_results8 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results8 = pd.concat([total_results8, up_result])
    print(len(total_results8))


# In[49]:


total_results8.reset_index(drop=True, inplace=True)


# # 가상현실

# In[50]:


query = '가상현실'


# In[52]:


total_results9 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results9 = pd.concat([total_results9, up_result])
    print(len(total_results9))


# In[53]:


total_results9.reset_index(drop=True, inplace=True)


# # gpt

# In[54]:


query = 'gpt'


# In[55]:


total_results10 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results10 = pd.concat([total_results10, up_result])
    print(len(total_results10))


# # 로봇

# In[90]:


query = '로봇'


# In[91]:


total_results11 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results11 = pd.concat([total_results11, up_result])
    print(len(total_results11))


# In[92]:


total_results11.reset_index(drop=True, inplace=True)


# # 3. 경제 (투자, 기업, 부자, 주식, 실업, 보험)

# # 투자

# In[93]:


query = '투자'


# In[95]:


total_results12 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results12 = pd.concat([total_results12, up_result])
    print(len(total_results12))
    


# In[96]:


total_results12.reset_index(drop=True, inplace=True)


# # 기업

# In[97]:


query = '기업'


# In[98]:


total_results13 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results13 = pd.concat([total_results13, up_result])
    print(len(total_results13))


# In[99]:


total_results13.reset_index(drop=True, inplace=True)


# # 부자

# In[100]:


query = '부자'


# In[101]:


total_results14 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results14 = pd.concat([total_results14, up_result])
    print(len(total_results14))


# In[102]:


total_results14.reset_index(drop=True, inplace=True)


# # 주식

# In[103]:


query = '주식'


# In[104]:


total_results15 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results15 = pd.concat([total_results15, up_result])
    print(len(total_results15))


# In[105]:


total_results15.reset_index(drop=True, inplace=True)


# # 실업

# In[116]:


query = '실업'


# In[117]:


total_results16 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results16 = pd.concat([total_results16, up_result])
    print(len(total_results16))


# In[118]:


total_results16.reset_index(drop=True, inplace=True)


# In[109]:


middle= pd.concat([total_results, total_results2,total_results3,total_results4,
                  total_results5,total_results6,total_results7,total_results8,
                  total_results9,total_results10,total_results11,total_results12,
                  total_results13,total_results14,total_results15,total_results16])


# In[111]:


# 데이터 수집 개수 중간 확인용
middle.shape


# # 보험

# In[119]:


query = '보험'


# In[120]:


total_results17 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results17 = pd.concat([total_results17, up_result])
    print(len(total_results17))


# In[121]:


total_results17.reset_index(drop=True, inplace=True)


# In[122]:


middle= pd.concat([total_results, total_results2,total_results3,total_results4,
                  total_results5,total_results6,total_results7,total_results8,
                  total_results9,total_results10,total_results11,total_results12,
                  total_results13,total_results14,total_results15,total_results16,
                  total_results17])


# In[124]:


middle.shape


# # 4. 환경 (날씨, 재활용)

# # 날씨

# In[129]:


query = '날씨'


# In[130]:


total_results18 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results18 = pd.concat([total_results18, up_result])
    print(len(total_results18))


# In[131]:


total_results18.reset_index(drop=True, inplace=True)


# # 재활용

# In[132]:


query = '재활용'


# In[134]:


total_results19 = pd.DataFrame()
for i in range(10):
    start=1+n_display*i
    print(start)
    result= getresult(client_id,client_secret,query,n_display,start,sort='sim')
    up_result = search_news_with_link(result)
    
    total_results19 = pd.concat([total_results19, up_result])
    print(len(total_results19))


# In[135]:


total_results19.reset_index(drop=True, inplace=True)


# # Concat

# In[136]:


crawl= pd.concat([total_results, total_results2,total_results3,total_results4,
               total_results5,total_results6,total_results7,total_results8,
              total_results9,total_results10,total_results11,total_results12,
               total_results13,total_results14,total_results15,total_results16,total_results17,
                 total_results18, total_results19])


# In[137]:


crawl.shape


# In[138]:


crawl.reset_index(drop=True, inplace=True)


# ## <font color = 'red'> 2) Text data preprocessing

# ## <font color= red> 2-1) cleaning

# #### content가 0인 행들은 네이터 연예 뉴스로써, 연예 뉴스에 해당하는 기사들을 제거하기 위해 0으로 추가하고, 삭제
# - 네이버 뉴스 기사들만 크롤링을 하고 싶어, dic_area에 해당하지 않는 기사 본문들은 0으로 content에 추가해주었습니다.  
# - 추가적으로 continue를 사용해서 해당하지 않는 id는 넘어가게 하고 싶었지만 해결이 잘 되지 않았고, 이에 따라 dictionary 형태로 변환할 때 title은 나왔지만, id가 없는 기사이기에 content에 빈 값으로 나와 dataframe이 만들어지지 않는 문제가 발생했습니다.  
# - 따라서 위와 같은 방식를 통해 0값으로 채우고 삭제하는 방향으로 전처리를 진행하였습니다.

# ### content == 0 행 제거

# In[139]:


cl= crawl.copy()


# In[140]:


cl.head()


# In[141]:


cl.drop(cl.loc[cl['content']==0].index, inplace=True)


# ## <font color=red> 2-2) 중복값 제거

# In[142]:


cl=cl.drop_duplicates()


# In[143]:


cl.reset_index(drop=True,inplace=True)


# In[144]:


cl[cl.duplicated(keep=False)]


# In[145]:


cl.shape


# ## 1차 dataframe 저장

# In[149]:


cl.to_csv('crawling_df.csv',index=False)


# ## 데이터 불러오기

# In[2]:


df=pd.read_csv('crawling_df.csv')


# In[3]:


df1=df.copy()


# ## <font color = red> 3) 1차 데이터 전처리 - 불용어, 어간 추출

# In[5]:


import copy
from konlpy.tag import Okt
import pykospacing
import kss
with open ('stopwords.txt','r',encoding='utf-8')as f:
    stopwords= f.readlines()
stopwords= [x.replace('\n','') for x in stopwords]
okt=Okt()


# In[6]:


df1.info()


# In[7]:


#교수님의 기존 stopwords.txt를 통해 1차적으로 불용어 제거 및 어간 추출 진행
def preprocess_korean(text):
    my_text=copy.copy(text)
    #\n 제거
    my_text = my_text.replace('\n','')
    spacer= pykospacing.Spacing() #띄어쓰기 교정
    my_text=spacer(my_text)
    sents=kss.split_sentences(my_text)
    
    p=re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]') #한글과 띄어쓰기를 제외한 모든 글자
    results=[]
    for sent in tqdm(sents):
        result=[]
        tokens= okt.morphs(sent, stem=True) #어간추출
        for token in tokens:
            token=p.sub('',token)
            if token not in stopwords: #stopwords에 없는 애들만 추가해라
                result.append(token) 
        results.extend(result) 
    result= ' '.join(results)
    
    return result  


# In[8]:


#대략 6일 정도 소요되었습니다
df1['preprocessing_content']= df1['content'].apply(lambda x: preprocess_korean(x))


# In[196]:


df1.to_csv('텍데분전처리1단계끝.csv',index=False)


# ## <font color= red> 4) 2차 불용어 제거 - 1차 전처리를 통해 나온 결과값들을 바탕으로 불용어 추가 수집

# In[227]:


df1=pd.read_csv('텍데분전처리1단계끝.csv')


# In[228]:


df=df1.copy()


# ###  <font color = red> 4-1) 1단계에서 전처리한 후, 본문이 없는 기사 행 삭제

# In[231]:


df.isnull().sum()


# In[232]:


#1차 전처리 후, 본문이 비어있는 행을 확인할 수 있음
df.iloc[[161]]


# In[233]:


df=df.dropna(axis=0)


# In[234]:


df.reset_index(inplace=True,drop=True)


# In[236]:


df.isnull().sum()


# In[237]:


len(df)


# ### <font color = red> 4-2) 1차 preprocessing 된 token들을 리스트에 저장하여 2차 불용어 처리 진행

# In[238]:


stopwords2= []
for i in tqdm(range(len(df))): #0~10377
    for value in list(df.preprocessing_content[i].split(' ')): 
        stopwords2.append(value) #값 추가 -> 10400개의 본문에 대한 토큰들을 stopwords2에 저장함


# In[239]:


imsi=pd.DataFrame(stopwords2)
imsi=imsi.rename(columns={0:'words'})


# In[240]:


#가장 많이 나온 불용어들을 뽑아서 새로운 불용어 리스트에 저장 (상위 30개 중, 조사 위주로)
word= pd.DataFrame(imsi['words'].value_counts()).head(30).index.tolist()


# In[241]:


# value_counts 값이 하나인 불용어들을 뽑아서 새로운 불용어 리스트에 저장 (하위 30개)
word2=pd.DataFrame(imsi['words'].value_counts()).tail(30).index.tolist()


# In[242]:


stopwords2_total=word+word2


# In[243]:


# 조사 아닌 단어들 중,의미가 있을법한 단어는 불용어에서 제거함
rm_set = {'대통령', '기업','기술','법','투자','서울'}
# 리스트 컴프리헨션 활용: 삭제할 원소 집합 데이터와 일일이 비교
stopwords2_total = [i for i in stopwords2_total if i not in rm_set]
print(stopwords2_total)


# In[244]:


# 추가적으로 기사에 할당된 토큰들을 직접 찾아 보면서 (약 150개의 기사들을 예시로 찾아봄) 공통적으로 필요없는 불용어들을 리스트에 추가해줌
stopwords3= ['',' ', '기자','무단','앙카라','로이터','뉴스','금지','무단','뉴스','제보','저자','방송','화면','캡처','사진','방송화면',
            '연합뉴스','왼쪽부터','데일리안','현지','시각','시간','기사내용','뉴시스','뉴스데스크','카카오톡','기','다리다','이메일',
            '앵커','자료조사','영상편집','리포트','채널','네이버','유튜브','구독','카카오','톡','전화','추가','영상','디자인',
            '페이스북','트위터','노컷뉴스','사이트','기사','내용','요약','출처','은','는','이','가','이다','하다','돼다','에','에서',
             '에선','라며','고','하','다','하고','하며','되다','뉴욕타임즈','오다','보다','따르다','가다','통해','에는','없다','대한',
             '때문','관련','경우','이르다','그렇다','에서는','뿐 아니다','지다','들다','대다','보이다','에도','이나','아니다',
             '씨','김','데','시','날','면서']


# In[245]:


# 조사들을 추가한 불용어 이외에 추가로 직접 찾은 불용어와의 비교를 통해 없는 불용어 추가
for i in stopwords3:
    if i not in stopwords2_total:
        stopwords2_total.append(i)


# In[246]:


stopwords2_total=' '.join(stopwords2_total)


# In[247]:


#https://junjun-94.tistory.com/18
#이 링크를 활용해 코드를 이해한 후, 함수로 변환하여 추가 불용어 처리 코드를 작성했습니다.
def preprocess_korean2(example):
    stop_words = stopwords2_total
    stop_words = stop_words.split(' ')

    word_tokens =word_tokenize(example)
    
    result=[]
    for w in tqdm(word_tokens):
        if w not in stop_words:
            result.append(w)
    
    result = ' '.join(result)
 

    return result


# In[248]:


df['preprocessing_content2']= df['preprocessing_content'].apply(lambda x: preprocess_korean2(x))


# In[249]:


df1=df.copy()


# In[250]:


# content에 내용이 없는 기사들 추가로 삭제
df1.query('preprocessing_content2==""')


# In[251]:


df1.drop([2844,4654,5608,5876,7782], axis=0,inplace=True)


# In[252]:


df1=df1.reset_index(drop=True)


# In[253]:


df1.to_csv('텍데분전처리2단계끝.csv')


# In[1]:


import pandas as pd


# In[4]:


pd.read_csv('data/텍데분전처리2단계끝.csv')['preprocessing_content2']


# In[27]:


pd.read_csv('data/텍데분전처리2단계끝.csv').iloc[:,1:].shape

