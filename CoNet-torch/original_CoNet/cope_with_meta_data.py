# -*- coding: utf-8 -*-
"""cope_with_meta_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15NAIpx9pQuVTAuWgwQrS4Uj4BRGEOtsz
"""


import os
import sys
w_dir = '/Users/linyishi/Desktop/毕业论文/dataset'
os.chdir(w_dir)

import pandas as pd
import numpy as np
import json
#meta_Books = pd.read_csv('meta_Books.csv')
#meta_Movies_and_TV = pd.read_csv('meta_Movies_and_TV.csv')
#meta_Movies_and_TV_used = pd.read_csv('meta_Movies_and_TV_used.csv',encoding='utf-8-sig')
#meta_books_used = pd.read_csv('meta_books_used.csv',encoding='utf-8-sig')

#books_df_sample = pd.read_csv('books_df_sample.csv')
#movies_df_sample = pd.read_csv('movies_df_sample.csv')
#books_df_sample.drop(columns='index',inplace=True)
#movies_df_sample.drop(columns='index',inplace=True)
#books_id = set(books_df_sample['itemId'])
#movies_id = set(movies_df_sample['itemId'])

#print(meta_Movies_and_TV.shape)
#print(meta_Books.shape)

#useful_columns=['asin','description','title']
#meta_Movies_and_TV_simp = meta_Movies_and_TV[useful_columns]
#meta_Books_simp = meta_Books[useful_columns]
#del meta_Movies_and_TV
#del meta_Books

#meta_Movies_and_TV_used = meta_Movies_and_TV_simp[meta_Movies_and_TV_simp.asin.isin(movies_id)].reset_index()
#meta_books_used = meta_Books_simp[meta_Books_simp.asin.isin(books_id)].reset_index()
#meta_Movies_and_TV_used.to_csv('meta_Movies_and_TV_used.csv',encoding='utf-8-sig',index=None)
#meta_books_used.to_csv('meta_books_used.csv',encoding='utf-8-sig',index=None)

#print(meta_Movies_and_TV_used.shape)
#print(meta_books_used.shape)

#meta_books_used.head(20)

movie_genre_path1 = 'MovieGenre.csv'
movie_genre_path2 = 'wiki_movie_plots_deduped.csv'
book_genre_path = 'booksummaries.txt'
movie_genre1 = pd.read_csv(movie_genre_path1,encoding='latin-1')
movie_genre2 = pd.read_csv(movie_genre_path2)
book_genre = pd.read_csv(book_genre_path,sep='\t',names=['WikipediaID','FreebaseID','title','author','pub_date','genres','summary'])

print(movie_genre1.columns) # 只有标题信息，没有简介信息
print(movie_genre2.columns) # 标题字段：Title 简介字段：Plot

print(movie_genre1.head(5))
print(movie_genre2.head(5))

movie_genre2['Genre'][:10]

"""### 统计一共有多少个genre"""

movie_genre_set_2 = list(movie_genre2['Genre'].unique())
#movie_genre_set_1 = list(movie_genre1['Genre'].unique())
book_genre_set = list(book_genre['genres'].unique())

movie_genre_set_2[:10]

movie_genres=set()
for x in movie_genre_set_2:
  try:
    gs=x.split('/')
    movie_genres=movie_genres.union(set(gs))
  except:
    print(x)
    pass

print(len(movie_genres))
print(movie_genres)

book_genres=set()
for x in book_genre_set:
  try:
    gs=set(eval(x).values())
    book_genres = book_genres.union(gs)
  except:
    print(x)
    pass

print(len(book_genres))
print(book_genres)

"""### 统计有多少电影/书籍能够匹配上"""

movies = [x.lower() for x in movie_genre1['Title'].tolist()]
books = [x.lower() for x in book_genre['title'].tolist()]

print(len(movies))
print(len(books))

#movies_r = meta_Movies_and_TV_used['title'].tolist()
#books_r = meta_books_used['title'].tolist()

#books_r[:100]

#print(len(movies_r))
#print(len(books_r))

#print(set(movies).intersection(movies_r))
#print(set(books).intersection(books_r))


'''
=======================================================================
先不管体裁这一特征如何匹配和训练，我先用不带标签训练出来的文本特征向量喂进去模型
=======================================================================
'''
# STEP1 粗糙地预处理文本成fasttext需要的格式 用什么文本？metadata的文本，先用

with open('test.txt','w',encoding='utf-8') as f:
  for i,row in book_genre.iterrows():
    outline = (row['summary'] + row['title']).replace("\t", " ").replace("\n", " ").replace('.',' ').replace('  ',' ')
    outline_list = outline.split(' ')
    outline = " ".join(outline_list)
    f.write(outline+'\n')
    f.flush()
  for i, row in movie_genre2.iterrows():
    outline = (row['Plot'] + row['Title']).replace("\t", " ").replace("\n", " ").replace('.', ' ').replace('  ',' ')
    outline_list = outline.split(' ')
    outline = " ".join(outline_list)
    f.write(outline + '\n')
    f.flush()
# STEP2 先用无监督的词向量嵌入来获取物品的向量表示
import fasttext
model = fasttext.train_unsupervised('test.txt')
model.save_model("test.bin")
model.get_input_vector() #获得原本句子的向量表示
model.get_sentence_vector() #获得新句子的向量表示
# STEP3 将物品的特征向量表示整合进模型，初步跑通结果
# 这里要跑的词向量，用两种商品元数据的描述去跑好了