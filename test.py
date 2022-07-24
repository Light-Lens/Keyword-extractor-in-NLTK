import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
print()

sentence = "Hello I'm Oxygen. I'm a natural language interface having access to all Horizon Security Networks, Defence Mechanism Systems and Srijan Srivastava's Private Server Protocols."

Lemmatizer = WordNetLemmatizer()
Stemmer = PorterStemmer()

stop_words = set(stopwords.words("english"))
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

TokenizeWordsWithoutStopwords = []
for word in tokens:
	if word not in stop_words:
		TokenizeWordsWithoutStopwords.append(word)

for word in TokenizeWordsWithoutStopwords:
	print(Stemmer.stem(word), Lemmatizer.lemmatize(word, "v"))

print(set(tokens) - set(TokenizeWordsWithoutStopwords), end="\n\n")
print(tokens, end="\n\n")
print(TokenizeWordsWithoutStopwords, end="\n\n")
print(stop_words)
print(tagged)
