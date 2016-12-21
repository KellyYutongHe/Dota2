
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from SentimentalDictionaryCreate import sentimental_dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm,cross_validation
from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
stemmer = SnowballStemmer("english")
stopword=set(stopwords.words('english'))


# In[3]:

#Mini Library
i_word = ["i","im","i'm","me","my","mine"]
u_word = ["u","you","ur", "ru", "your","you're","youre"]
he_word = ["he","him","his","hes","he's"]
she_word = ["she","her","hers","shes","she's"]
it_word = ["it","its","it's"]
we_word = ["we","our","ours"]
they_word = ["they","them","their","theirs"]
common_words = [i_word,u_word,he_word,she_word,it_word,we_word,they_word]


# In[4]:

# chat data
chat = pd.read_csv('chat.csv')
# region data
cluster = pd.read_csv('cluster_regions.csv')
# match data include wins, first blood time, 
match = pd.read_csv('match.csv')
team_fights = pd.read_csv('teamfights.csv')
#team_fights_player = pd.read_csv('c:/Python27/teamfights_players.csv')
match_outcomes = pd.read_csv('match_outcomes.csv')


# In[5]:

match_cluster_merge = match.merge(cluster,on='cluster')
chat_region_merge = chat.merge(match_cluster_merge[['match_id','region']],on = 'match_id')
# We select the game region which speack english
country = ['SINGAPORE','US WEST','US EAST','AUSTRALIA','EUROPE']
# We get the new related data set
chat_region_merge = chat_region_merge[chat_region_merge.region.isin(country)]


# In[6]:

chat_region_merge.head(15)


# In[7]:

#chat data with match result
match_results = match[['match_id','radiant_win']]
chat_data = chat_region_merge.merge(match_results,on='match_id')
chat_data['key'] = chat_data['key'].str.lower()
chat_data['key'] = chat_data['key']
#chat log with match ID and key
win_log = chat_data[((chat_data.radiant_win==1) & (chat_data.slot <=4)) | ((chat_data.radiant_win==0) & (chat_data.slot > 4))][['match_id','key']]
lost_log = chat_data[(chat_data.radiant_win==0) & (chat_data.slot <=4)| ((chat_data.radiant_win==1) & (chat_data.slot > 4))][['match_id','key']]
#win&lost team chat data
win_chat = chat_data[((chat_data.radiant_win==1) & (chat_data.slot <=4)) | ((chat_data.radiant_win==0) & (chat_data.slot > 4))].key.values
lost_chat = chat_data[(chat_data.radiant_win==0) & (chat_data.slot <=4)| ((chat_data.radiant_win==1) & (chat_data.slot > 4))].key.values


# In[8]:

chat_data['key']


# In[9]:

#average sentence count
print(win_log.match_id.value_counts().mean())
print(lost_log.match_id.value_counts().mean())


# In[10]:

#average character count - win team
win_length = []
for keyword in win_log['key']:
    win_length.append(len(str(keyword)))
win_log['length'] = win_length
print(win_log.length.mean())
win_log.head()


# In[11]:

#average character count - lost team
lost_length = []
for keyword in lost_log['key']:
    lost_length.append(len(str(keyword)))
lost_log['length'] = lost_length
print(lost_log.length.mean())
lost_log.head()


# In[12]:

#NLP


# In[13]:

#remove unicode
ascii = set(string.printable)   
def remove_non_ascii(s):
    return filter(lambda x: x in ascii, s)


# In[14]:

#main NLP function
#parameter: a chat list - win&lost team chat data
#return: a word list without stop words
def nlprocess(chatList):
    chat = ' '.join(map(str,chatList))
    chat = remove_non_ascii(chat)
    chat_text = chat.translate(None, string.punctuation)
    chat_word = word_tokenize(chat_text)
    count = 0
    for key in chat_word:
        if not key.endswith('e'):
            chat_word[count] = str(stemmer.stem(key))
            if chat_word[count].endswith('i') and chat_word[count] != "i" and chat_word[count] != "hi":
                chat_word[count] = chat_word[count].replace(chat_word[count][len(chat_word[count])-1], 'y')
            if "haha" in chat_word[count]:
                chat_word[count] = "haha"
            if "fck" in chat_word[count]:
                chat_word[count] = "fuck"
        for wordlist in common_words:
            for commonword in wordlist:
                if chat_word[count] == commonword:
                    chat_word[count] = "a"
        count = count+1
    chat_stopfreewords = [word for word in chat_word if word not in stopword]
    return chat_stopfreewords


# In[15]:

#function that generates the most frequent 100 words from a word list
#parameter: a word list
#return: a list of top 100 frequent word in the word list
def generateFrequent100(wordList):
    chat_freq = nltk.FreqDist(wordList)
    chat_100_words = zip(*chat_freq.most_common(100))[0]
    chat_100_words = ' '.join(map(str,chat_100_words))
    return chat_100_words


# In[16]:

#function that generates a word cloud from a word list
def generateWordCloud(wordList):
    wordcloudword = ' '.join(map(str,wordList))
    wordcloud = WordCloud().generate(wordcloudword)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# In[17]:

#create the most frequent 100 words in win team chat
win_stopfreewords = nlprocess(win_chat)
win_100_words = generateFrequent100(win_stopfreewords)
win_100_words


# In[18]:

win_stopfreewords


# In[19]:

#generate the word cloud for win team
generateWordCloud(win_stopfreewords)


# In[20]:

#create the most frequent 100 words in lost team chat
lost_stopfreewords = nlprocess(lost_chat)
lost_100_words = generateFrequent100(lost_stopfreewords)
lost_100_words


# In[21]:

lost_stopfreewords


# In[22]:

#generate the word cloud for lost team
generateWordCloud(lost_stopfreewords)


# In[23]:

#create the most frequent 100 words in all chat data
all_text = []
for chat in chat_region_merge.key:
    all_text.append(str(chat).lower())

all_stopfreewords = nlprocess(all_text)
all_100_words = generateFrequent100(all_stopfreewords)
all_100_words


# In[24]:

all_stopfreewords


# In[25]:

#generate the word cloud for all chat data
generateWordCloud(all_stopfreewords)


# In[23]:

# all_words = nltk.FreqDist(text)
# all_words.most_common(10)[1]
# new = zip(*all_words.most_common(100))
# word = new[0]
# freq = new[1]
# all_words.most_common(100)


# In[24]:

#Sentimental Test


# In[25]:

def sentimentalTest(str):
    analyzer = SentimentIntensityAnalyzer()
    if str in sentimental_dict.keys():
        return sentimental_dict[str]
    else:
        return analyzer.polarity_scores(str)['compound']

sentimentalTest("ez")


# In[ ]:

win_sentimental_score = 0
for word in win_stopfreewords:
    win_sentimental_score += sentimentalTest(word)
win_sentimental_score /= len(win_stopfreewords)
print(win_sentimental_score)


# In[ ]:

lost_sentimental_score = 0
for word in lost_stopfreewords:
    lost_sentimental_score += sentimentalTest(word)
lost_sentimental_score /= len(lost_stopfreewords)
print(lost_sentimental_score)


# In[25]:

#Classification


# In[26]:

#chat log with class labels
win_group = win_log.groupby('match_id')
chat_content_win = []
for i in win_group.size().index:
    chat_content_win.append(win_group.get_group(i).key.values)

lost_group = lost_log.groupby('match_id')
chat_content_lost = []
for i in lost_group.size().index:
    chat_content_lost.append(lost_group.get_group(i).key.values)


# In[27]:

new_df_win = pd.DataFrame(zip(chat_content_win),columns=['keys'])
new_df_win['win'] = 1

new_df_lost = pd.DataFrame(zip(chat_content_lost),columns=['keys'])
new_df_lost['win'] = 0

new_df = pd.concat([new_df_win, new_df_lost])
new_df


# In[28]:

#Bag-of-words
nlped_chat = []
for key in new_df["keys"]:
    nlped_chat.append(" ".join(nlprocess(key)))
nlped_chat


# In[29]:

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 1000)
data_features = vectorizer.fit_transform(nlped_chat).toarray()
vocab = vectorizer.get_feature_names()
print(vocab)


# In[30]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_features, new_df["win"], test_size=0.7)


# In[34]:

clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
print("Trainning completed")
accuracy_svm = clf_svm.score(X_test, y_test)
print(accuracy_svm)


# In[35]:

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Trainning completed")
accuracy_nb = gnb.score(X_test, y_test)
print(accuracy_nb)


# In[36]:

# clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,))
# clf_mlp.fit(X_train, y_train)
# print("Trainning completed")
# accuracy_mlp = clf_mlp.score(X_test, y_test)
# print(accuracy_mlp)
print(y_test.mean())


# In[37]:

forest = RandomForestClassifier(n_estimators = 1000) 
forest = forest.fit(X_train, y_train)
print("Trainning completed")
accuracy_rf = forest.score(X_test, y_test)
print(accuracy_rf)


# In[ ]:

clf_svm_linear = svm.SVC(kernel = 'linear')
clf_svm_linear.fit(X_train, y_train)
print("Trainning completed")
accuracy_svm_linear = clf_svm_linear.score(X_test, y_test)
print(accuracy_svm_linear)


# In[77]:

#tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
new_data_features = transformer.fit_transform(data_features).toarray()

print(new_data_features.shape)


# In[78]:

new_X_train, new_X_test, new_y_train, new_y_test = cross_validation.train_test_split(new_data_features, new_df["win"], test_size=0.4)


# In[41]:

clf_svm = svm.SVC()
clf_svm.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_svm = clf_svm.score(new_X_test, new_y_test)
print(accuracy_svm)


# In[79]:

gnb = GaussianNB()
gnb.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_nb = gnb.score(new_X_test, new_y_test)
print(accuracy_nb)


# In[ ]:

clf_svm_linear = svm.SVC(kernel = 'linear')
clf_svm_linear.fit(new_X_train, new_y_train)
print("Trainning completed")
accuracy_svm_linear = clf_svm_linear.score(new_X_test, new_y_test)
print(accuracy_svm_linear)


# In[33]:

total_data_features = vectorizer.fit_transform([" ".join(win_stopfreewords)," ".join(lost_stopfreewords)]).toarray()
print(total_data_features.shape)


# In[80]:

new_total_data_features = transformer.fit_transform(total_data_features).toarray()
new_total_data_features


# In[81]:

remove_indecies = []
for index in range(len(new_total_data_features[0])):
    if (new_total_data_features[0][index] < 0.01) and (new_total_data_features[1][index] < 0.01):
        remove_indecies.append(index)
print(len(remove_indecies))
print(new_data_features.shape)
new_data_features = np.delete(new_data_features, remove_indecies, 1)
print(new_data_features.shape)
# new_total_data_features


# In[83]:

total_X_train, total_X_test, total_y_train, total_y_test = cross_validation.train_test_split(new_data_features, new_df["win"], test_size=0.4)
gnb = GaussianNB()
gnb.fit(total_X_train, total_y_train)
print("Trainning completed")
accuracy_nb = gnb.score(total_X_test, total_y_test)
print(accuracy_nb)


# In[84]:

clf_svm = svm.SVC()
clf_svm.fit(total_X_train, total_y_train)
print("Trainning completed")
accuracy_svm = clf_svm.score(total_X_test, total_y_test)
print(accuracy_svm)


# In[ ]:

forest = RandomForestClassifier(n_estimators = 1000) 
forest = forest.fit(total_X_train, total_y_train)
print("Trainning completed")
accuracy_rf = forest.score(total_X_test, total_y_test)
print(accuracy_rf)


# In[1]:

clf_svm_linear = svm.SVC(kernel = 'linear')
clf_svm_linear.fit(total_X_train, total_y_train)
print("Trainning completed")
accuracy_svm_linear = clf_svm_linear.score(total_X_test, total_y_test)
print(accuracy_svm_linear)

