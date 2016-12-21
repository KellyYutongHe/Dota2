
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from os import path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import *
from nltk.corpus import stopwords
from sklearn import *
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
tokenizer = word_tokenize
import string
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from wordcloud import WordCloud
stemmer = SnowballStemmer("english")
stopword=set(stopwords.words('english'))
wnl = WordNetLemmatizer()
#remove unicode
ascii = set(string.printable)   
def remove_non_ascii(s):
    return filter(lambda x: x in ascii, s)
#from SentimentalDictionaryCreate import sentimental_dict


# In[2]:

# #Mini Library
# i_word = ["i","im","i'm","me","my","mine"]
# u_word = ["u","you","ur", "ru", "your","you're","youre"]
# he_word = ["he","him","his","hes","he's"]
# she_word = ["she","her","hers","shes","she's"]
# it_word = ["it","its","it's"]
# we_word = ["we","our","ours"]
# they_word = ["they","them","their","theirs"]
common_words = ['im','u','ur','ru','youre','hes','shes','its']


# In[3]:

# chat data
chat = pd.read_csv('chat.csv')
# region data
cluster = pd.read_csv('cluster_regions.csv')
# match data include wins, first blood time, 
match = pd.read_csv('match.csv')
team_fights = pd.read_csv('teamfights.csv')
# team_fights_player = pd.read_csv('c:/Python27/teamfights_players.csv')
match_outcomes = pd.read_csv('match_outcomes.csv')

match_cluster_merge = match.merge(cluster,on='cluster')
chat_region_merge = chat.merge(match_cluster_merge[['match_id','region']],on = 'match_id')
# We select the game region which speack english
country = ['SINGAPORE','US WEST','US EAST','AUSTRALIA','EUROPE']
# We get the new related data set
chat_region_merge = chat_region_merge[chat_region_merge.region.isin(country)]


# # Simple basic data analysis

# Total chat counts analysis match level

# In[ ]:

chat_size = chat_region_merge.match_id.value_counts()
chat_size.describe()


# In[ ]:

plt.hist(chat_size.values,bins=np.unique(chat_size.values))
plt.ylabel('Frequency')
plt.xlabel('Chat Count in One Match')
plt.title('Chat Counts Distribution')
plt.show()
plt.hist(chat_size.values,bins=50,range=(0,150))
plt.ylabel('Frequency')
plt.xlabel('Chat Count in One Match')
plt.title('Chat Counts Distribution')


# In[ ]:

p,n = np.histogram(chat_size.values,bins=np.unique(chat_size.values),density=True)
n = n[:-1]
plt.plot(n,p,marker='o',linestyle='none')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Chat Count in One Match(log)')
plt.title('Chat Counts Distribution In Log Scale')


# Neither normal or plower law distributed, heavily skew to the right

# Total chat counts analysis individual level
# 

# In[ ]:

player_size = chat_region_merge.groupby(['match_id','unit']).size()


# In[ ]:

p,n = np.histogram(player_size.values,bins=np.unique(player_size.values),density=True)
n = n[:-1]
plt.plot(n,p,marker='o',linestyle='none')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Chat Count Per Person(log)')
plt.title('Chat Counts Distribution In Log Scale')


# For players, chat counts of players tend to more like powerlaw distributed.

# # Win and Lost analysis

# In[4]:

def win_lost_seperate(df):
    win_df = df[((df.radiant_win==1) & (df.slot <=4)) | ((df.radiant_win==0) & (df.slot > 4))]
    lost_df = df[((df.radiant_win==0) & (df.slot <=4)) | ((df.radiant_win==1) & (df.slot > 4))]
    return win_df, lost_df


# In[5]:

chat_data = chat_region_merge.merge(match[['match_id','radiant_win']],on='match_id')
win_df, lost_df = win_lost_seperate(chat_data)


# In[ ]:

win_size = win_df.match_id.value_counts()
lost_size = lost_df.match_id.value_counts()
win_size.describe()


# In[ ]:

lost_size.describe()


#  No significance between win and lost sentences counts, no signs that win team tend to speak more or the lost team

# In[6]:

# NLTK words cleanning.
chat_data['key'] = chat_data['key'].str.lower()
for i in range(0,len(chat_data['key'].values)):
    sentence = remove_non_ascii(str(chat_data['key'].values[i]).translate(None, string.punctuation)).split()
    for k in range(0,len(sentence)):
        word = ''.join(set(sentence[k])) if len(sentence[k]) >= 8 else sentence[k]
        sentence[k] = '' if word.isdigit() else wnl.lemmatize(word) if wnl.lemmatize(word).endswith(('e','y')) else 'haha' if set(word) <= set('ha') else 'fuck' if set(word) <= set('fuck') else 'gg' if set(word) <= set('ggwp') else str(stemmer.stem(word))
        sentence[k] = '' if sentence[k] in stopword else '' if sentence[k] in common_words else sentence[k]
    chat_data['key'].values[i] = ' '.join(filter(None, sentence))
chat_data = chat_data[chat_data.key.values != '']


# Wordcloud Generating and Top 100 words

# In[ ]:

win_df, lost_df = win_lost_seperate(chat_data)


# In[ ]:

def tokenizer(word_list):
    return word_tokenize(' '.join(word_list))


# In[ ]:

win_word = tokenizer(win_df.key.values)
lost_word = tokenizer(lost_df.key.values)


# In[ ]:

#function that generates the most frequent 100 words from a word list
#parameter: a word list
#return: a list of top 100 frequent word in the word list
def generateFrequent100(wordList):
    chat_freq = FreqDist(wordList)
    chat_100_words = zip(*chat_freq.most_common(100))[0]
    chat_100_words = ' '.join(map(str,chat_100_words))
    return chat_100_words


# In[ ]:

#function that generates a word cloud from a word list
dota_mask = np.array(Image.open(path.join("DOTA2.jpg")))
def generateWordCloud(wordList):
    wordcloudword = ' '.join(map(str,wordList))
    wc = WordCloud(background_color="black", max_words=150, mask=dota_mask)
    wordcloud = wc.generate(wordcloudword)
    plt.figure(figsize=(8,8))
    plt.imshow(wc)
    plt.axis("off")


# In[ ]:

win_100_words = generateFrequent100(win_word)
generateWordCloud(win_word)


# In[ ]:

lost_100_words = generateFrequent100(lost_word)
generateWordCloud(lost_word)


# # Different Time Period Analysis

# In[7]:

# Set up for picking chat basing on different time period.
chat_group = chat_data.groupby('match_id')
team_fights = team_fights[team_fights.match_id.isin(chat_group.size().index)]
team_group = team_fights.groupby('match_id')


# In[8]:

# Pick out data according to time interval
time_interval = []
for i in team_group.size().index:
    time_interval.append((team_group.get_group(i).end.values[-1]) * 3/4)    
df_time = pd.DataFrame(zip(team_group.size().index,time_interval),columns=['match_id','begin_time'])
chat_data = chat_data.merge(df_time,on='match_id')
chat_data = chat_data[chat_data.time >= chat_data.begin_time]


# In[9]:

# Get win & lost dataframe
win_data, lost_data = win_lost_seperate(chat_data)


# In[ ]:

win_word2 = tokenizer(win_data.key.values)
lost_word2 = tokenizer(lost_data.key.values)


# In[ ]:

win_100_words = generateFrequent100(win_word2)
generateWordCloud(win_word2)


# In[ ]:

lost_100_words = generateFrequent100(lost_word2)
generateWordCloud(lost_word2)


# In[ ]:

#function that generates a word cloud from a word list
def generateWordCloud1(wordList):
    wordcloudword = ' '.join(map(str,wordList))
    wc = WordCloud(max_font_size=40)
    wordcloud = wc.generate(wordcloudword)
    plt.figure(figsize=(8,8))
    plt.imshow(wc)
    plt.axis("off")


# In[ ]:

generateWordCloud1(win_word)


# In[ ]:

generateWordCloud1(lost_word)


# In[ ]:

generateWordCloud1(win_word2)


# In[ ]:

generateWordCloud1(lost_word2)


# In[ ]:

long_list = []
for i in FreqDist(win_word).keys():
    if len(i) >=9:
        long_list.append(i)


# In[ ]:

long_list


# Basing on the result we can see that, if we set word length longer than 9, nearlly all the words in the list is just short words with duplicated characters, in that situation it is safe for us to short them conservativly by taking the non-duplicated character and regroup to a shorter word, even it is meaningful words, after we take the non-duplicated character and regroup, these words don't lose lot of information. It help us a lot to make data cleanner and efficient in classification process to reduce number of features. Because even with words longer than 9, we have 10097 of them.

# # Classification and Prediction

# First we need new df with label of win as 1 and lost as 0

# In[ ]:

# We first analyze whole time interval match

win_group = win_df[['match_id', 'key']].groupby('match_id')
lost_group = lost_df[['match_id', 'key']].groupby('match_id')
win_content = [' '.join(win_group.get_group(i).key.values.tolist()) for i in win_group.size().index]
lost_content = [' '.join(lost_group.get_group(i).key.values.tolist()) for i in lost_group.size().index]

new_df_win = pd.DataFrame(win_content,columns=['keys'])
new_df_win['win'] = 1
new_df_lost = pd.DataFrame(lost_content,columns=['keys'])
new_df_lost['win'] = 0
new_input = pd.concat([new_df_win, new_df_lost])


# In[10]:

# We first analyze whole time interval match

win_group = win_data[['match_id', 'key']].groupby('match_id')
lost_group = lost_data[['match_id', 'key']].groupby('match_id')
win_content = [' '.join(win_group.get_group(i).key.values.tolist()) for i in win_group.size().index]
lost_content = [' '.join(lost_group.get_group(i).key.values.tolist()) for i in lost_group.size().index]

new_df_win = pd.DataFrame(win_content,columns=['keys'])
new_df_win['win'] = 1
new_df_lost = pd.DataFrame(lost_content,columns=['keys'])
new_df_lost['win'] = 0
new_input = pd.concat([new_df_win, new_df_lost])


# In[11]:

# Generate bag of words.
feature_input = new_input['keys'].values.tolist()


# In[12]:

vectorizer = CountVectorizer(analyzer = "word",max_features = 500)
data_features = vectorizer.fit_transform(feature_input).toarray()


# In[ ]:

vocab = vectorizer.get_feature_names()


# In[13]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_features, new_input["win"], test_size=0.70)


# In[14]:

clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
print("Trainning completed")
accuracy_svm = clf_svm.score(X_test, y_test)
print(accuracy_svm)


# In[15]:

clf_svm_linear = svm.SVC(kernel = 'linear')
clf_svm_linear.fit(X_train, y_train)
print("Trainning completed")
accuracy_svm_linear = clf_svm_linear.score(X_test, y_test)
print(accuracy_svm_linear)


# In[16]:

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Trainning completed")
accuracy_nb = gnb.score(X_test, y_test)
print(accuracy_nb)


# In[17]:

forest = RandomForestClassifier(n_estimators = 1000) 
forest = forest.fit(X_train, y_train)
print("Trainning completed")
accuracy_rf = forest.score(X_test, y_test)
print(accuracy_rf)


# In[18]:

#tf-idf
transformer = TfidfTransformer(smooth_idf=False)
new_data_features = transformer.fit_transform(data_features).toarray()


# In[19]:

new_X_train, new_X_test, new_y_train, new_y_test = cross_validation.train_test_split(new_data_features, new_input["win"], test_size=0.4)


# In[20]:

clf_svm = svm.SVC()
clf_svm.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_svm = clf_svm.score(new_X_test, new_y_test)
print(accuracy_svm)


# In[21]:

gnb = GaussianNB()
gnb.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_nb = gnb.score(new_X_test, new_y_test)
print(accuracy_nb)


# In[22]:

forest = RandomForestClassifier(n_estimators = 1000) 
forest = forest.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_rf = forest.score(new_X_test, new_y_test)
print(accuracy_rf)


# In[23]:

clf_svm_linear = svm.SVC(kernel = 'linear')
clf_svm_linear.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_svm_linear = clf_svm_linear.score(new_X_test, new_y_test)
print(accuracy_svm_linear)


# In[ ]:

chat_data


# In[ ]:



