#!/usr/bin/env python
# coding: utf-8

# #upload libraries
import sqlite3
import pandas as pd
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from itertools import islice
import re
import h5py
from gensim.corpora import Dictionary
from keras.utils import np_utils
from helper_text import main_clean
#from model import load_model
from keras.models import load_model

model = load_model("Final_weights/final_model.h5")


#-----------------Example Testing-------------------------
print('Example: ')
twt = 'Trump is a great president. MAGA. Buildthewall.'
print(twt)

x, x_s = main_clean(twt,19)
#predict

print(model.summary())

pro_pol = model.predict([x, x_s])[:,1]
print('the probability this tweet is pro Trump is ', pro_pol)


#-----------------Large Scale Testing----------------------
print('testing ...')

test_X_s = np.load("test_X_s.npy")
test_X=np.load("test_X.npy")
test_Y=np.load("test_Y.npy")
dictionary = Dictionary.load('Dictionary/dic.txt')
dictionary_s = Dictionary.load('Dictionary/dic_s.txt')
dictionary_size=len(dictionary)
dictionary_size_s=len(dictionary_s)
test_pred = model.predict([test_X, test_X_s])
test_pred = (test_pred[:,0] <= 0.5).astype(int)


score = model.evaluate( [test_X, test_X_s] , np_utils.to_categorical(test_Y, 2))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# #get list of files in our directory so we can loop through them 
"""
path = '../../Consolidate_Oct28/Brexit_snapshot5.db'
conn=sqlite3.connect(path)
c=conn.cursor()


def update(users, tweets):
	bar = progressbar.ProgressBar()
	for twt in bar(tweets):
		id_ = twt[0]
		u_id = twt[1]
		u = dict()
		u['id'] = twt[1]
		u['screen_name'] = twt[2]
		u['name'] = twt[3]
		u['description'] = twt[4]
		u['n_followers'] = twt[5]
		u['n_friends'] = twt[6]
		u['n_tweets_user'] = twt[7]
		u['location'] = twt[8]
		u['created_at'] = twt[9]
		# #here we want to keep track of how many tweets are used to compute the polarity
		try:
			u['n_tweets_model'] = users[u_id]['n_tweets_model'] + 1
		except Exception as e:
			#print(e)
			u['n_tweets_model'] = 1
		#get tweet
		x, x_s = main_clean(twt[-1])
		p = model.predict([x, x_s])[:,1][0]
		#we want to update polarity
		try:
			n = u['n_tweets_model']
			u['polarity'] = (n-1)/n * users[u_id]['polarity'] + 1/n * p
		except Exception as e:
			#print(e)
			u['polarity'] = p
			#u['polarity'] = [p]
		users[u_id] = u
	return(users)


all_tweets = c.execute('SELECT tweet.tweet_id,tweet.user_id,user_profile.screen_name, user_profile.name, user_profile.description, user_profile.followers_count,user_profile.friends_count,user_profile.statuses_count,user_profile.location, user_profile.created_at,tweet.text FROM tweet INNER JOIN user_profile ON tweet.user_id=user_profile.user_id').fetchall()


#define dictionary where users are keys 
users =dict()
count = 0. #to track how many batches of data we visited 
n = 5000  # Or whatever chunk size you want
batches = int(len(all_tweets)/n)

for b in range(batches):
	tweet_batch = all_tweets[(b-1)*n:b*n]
	count += 1
	print(count)
	users = update(users, tweet_batch)
	print('user count ',len(users))

print('number of users in total: ', len(users))
"""