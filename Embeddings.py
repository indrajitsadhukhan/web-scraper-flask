# pip install requests
# pip install re
# pip install bs4
# pip install collections
# pip install html
# pip install openai
# pip install tiktoken

import requests
import re
from urllib.request import urljoin

from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import openai
import numpy as np
import tiktoken
import os
from dotenv import load_dotenv,find_dotenv
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse


from openai.embeddings_utils import distances_from_embeddings

max_tokens = 500
upper_limit = 10
count=0


"""
Functions
1. split_into_many
2. depth0
3. crawl
4. text_csv
5. openai_embeddings
6.  remove_newlines
7. create_context
8. answer_question
9. getData
10. scrape
"""

API_ENDPOINT="https://sravanakumar13sathish.openai.azure.com/"

OPENAI_API_KEY="f93979cbf9894257affd4fee8b4e08fb"

COMPLETION_MODEL="Policy_GPT"
CHAT_COMPLETION_MODEL="Chat"
EMBEDDING_MODEL="text-embedding-ada-002"


df = pd.DataFrame()
# Use Azure OpenAI
openai.api_key=OPENAI_API_KEY
openai.api_type='azure'
openai.api_base=API_ENDPOINT
api_version_gpt35 = "2023-03-15-preview"
openai.api_version=api_version_gpt35
lvl=0
count=0


BASE_URL=""

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

    # Method for crawling a url at next level
def level_crawler(input_url):
    links_intern = set()
    depth = 2
    encoding = "utf-8"
    current_url_domain = urlparse(input_url).netloc


    global lvl,count
    lvl+=1
    temp_urls = set()
    current_url_domain = urlparse(input_url).netloc


    # Creates beautiful soup object to extract html tags
    beautiful_soup_object = BeautifulSoup(
        requests.get(input_url).content, "lxml")

    # Access all anchor tags from input
    # url page and divide them into internal
    # and external categories
    for anchor in beautiful_soup_object.findAll("a"):
        href = anchor.attrs.get("href")
        if(href != "" or href != None):
            href = urljoin(input_url, href)
            href_parsed = urlparse(href)
            href = href_parsed.scheme
            href += "://"
            href += href_parsed.netloc
            href += href_parsed.path
            final_parsed_href = urlparse(href)
            is_valid = bool(final_parsed_href.scheme) and bool(
                final_parsed_href.netloc)
            if is_valid:
                # if current_url_domain not in href and href not in links_extern:
                # 	print("Extern - {}".format(href))
                # 	links_extern.add(href)
                if href.startswith(input_url) and href not in links_intern:
                    count+=1
                    if(count>=upper_limit):
                        return []
                    print("Link %d: "%(count) + href)
                    links_intern.add(href)
                    # Extract text from href and store it in a text_scraped folder.
                    page = urlopen(href)
                    encoding = "utf-8"
                    htmlcontent = (page.read()).decode(encoding)
                    soup = BeautifulSoup(htmlcontent,"html.parser")
                    with open("text"+"/depth%d_%d.txt"%(lvl,count),'w',encoding=encoding,errors="ignore") as f:
                        f.write(href+"\n"+remove_lines(soup.get_text()))
                    temp_urls.add(href)
    return temp_urls



def new_crawl(input_url,depth=2):
    # Set for storing urls with same domain
    if(os.path.exists("text")==False):
        os.mkdir("text")

    if(depth == 0):
        print(input_url)

    elif(depth == 1):
        lvl=0
        level_crawler(input_url)

    else:
        # We have used a BFS approach
        # considering the structure as
        # a tree. It uses a queue based
        # approach to traverse
        # links upto a particular depth.
        
        queue = []
        queue.append(input_url)
        for j in range(depth):
            for count in range(len(queue)):
                url = queue.pop(0)
                urls = level_crawler(url)
                for i in urls:
                    queue.append(i)


def depth0(url):
    try:
        url_text=[]
        page = urlopen(url)
        htmlcontent = (page.read()).decode("latin1")
        soup = BeautifulSoup(htmlcontent,"html.parser")
        # Loop through all the hyperlinks present in the HTML and if we get http at the begining we add them to a list
        for link in soup.find_all('a'):
            h=link.get('href')
            if h and h.startswith('http'):
                url_text.append(link.get('href'))
        return url_text,soup.get_text()
    except:
        print("Failed to do depth0 scraping.")
        return [],""

def check_domain(url):
    domain = urlparse(url).netloc
    if(domain ==""):
        return True
    if(domain != urlparse(BASE_URL).netloc):
        return False
    return True

def depth1(url):
    urls,mainpage_content = depth0(url)
    print("Number of links for Depth=1: ",len(urls))
    depth1_urls=[]
    for c,link in enumerate(urls):
        if(check_domain(link)==False):
            continue
        # if(c>upper_limit):
        #     break
        text_hyperlink_list,hyperlink_content = depth0(link)
        for link_text in text_hyperlink_list:
            depth1_urls.append(link_text)
        print("Link %d: "%(c+1),link)
        with open("text/depth1_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth1 scraping done!")
    return depth1_urls

def depth2(url):
    depth1_urls= depth1(url)
    
    print("Number of links for Depth=2:",len(depth1_urls))
    depth2_urls=[]
    for c,link in enumerate(depth1_urls):
        if(check_domain(link)==False):
            continue
        text_hyperlink_list,hyperlink_content = depth0(link)
        for text_link in text_hyperlink_list:
            depth2_urls.append(text_link)
        print("Link %d: "%(c+1),link)
        with open("text/depth2_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth2 scraping done!")
    return depth2_urls

def crawl(url):
    BASE_URL=url
    depth=2
    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")
    if depth==0:
        depth1_urls,mainpage_content = depth0(url)
        with open("text/depth_0.txt",'w',encoding="latin1",errors='ignore') as f:
            f.write(mainpage_content)
        print("Depth0 scraping done!")
    if depth==1:
       depth1(url)
    if depth==2:
        depth2(url)

    text_csv()

# Create a CSV file from all the text files in text/
def text_csv():
    texts=[]
        # Create a directory to store the text files
    if not os.path.exists("processed/"):
            os.mkdir("processed/")

    # Get all the text files in the text directory
    for file in os.listdir("text/"):
        # Open the file and read the text
        with open("text/"+ file, "r", encoding="latin-1") as f:
            text = f.read()
            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])
    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    df.head()
    openai_embeddings()


def openai_embeddings():
    # Tiktoken is used to compute the number of tokens.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    shortened = []
    for row in df.iterrows():
        if row[1]['text'] is None:
            continue
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        else:
            shortened.append( row[1]['text'] )
        df = pd.DataFrame(shortened, columns = ['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        df.n_tokens.hist()

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    df.to_csv('processed/embeddings.csv')
    df.head()



def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def remove_lines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    return serie

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    print("Creating context...")
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    print("Q-embeddings")
    # Get the distances from the embeddings

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    # print(df['embeddings'])
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    # print(df['distances'])

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])
    
    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model=CHAT_COMPLETION_MODEL,
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1000,
    stop_sequence=None
):
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    print("CHK-12")
    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            messages=[{"role":"assistant","content":f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"},{"role":"user","content":"Question:"+question}],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            engine=model,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""



def askQuestion(question):
    print("Question: "+question)
    print("Answer Question called.")
    answer=answer_question(df,question)
    print("Answer:"+answer)
    return answer


def scrape(url):
    print(url)
    new_crawl(url)
    text_csv()
    return 200
