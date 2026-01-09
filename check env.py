import pandas, numpy, matplotlib, seaborn, nltk, sklearn, wordcloud
print("✅ All libraries are working!")
import pandas as pd
import numpy as np

import seaborn as sns
# Pandas
import pandas as pd

# Numpy
import numpy as np

# Matplotlib + Seaborn (for graphs)
import matplotlib.pyplot as plt
import seaborn as sns

# Nltk
import nltk
nltk.download('stopwords') # downloads stopwords (used later)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Scikit learn (models and visualization tools)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# Wordcloud for some extra data visualization!
from wordcloud import WordCloud

# This will be used for Reading and Writing our model from files
import pickle


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
## Eg: Let's create a table of names and ages

# Let's take a simple list of values:
values = [1, 3, 2, 5]

# Printing: Not too bad, but this would be hard to follow as our list grows
print("Printing the list:", values)
print()

print("Matplotlib graph of the list:")
# Plotting the list:
plt.plot(values)
plt.title("Line plot of values") # title of our graph
plt.show()                       # telling python to render our graph

## Eg: Remove common english word "fluff" from the following sentence
sentence = "This is a simple sentence demonstrating stopwords removal."
words = sentence.split()

# Manual approach:
# Defining set containing all stopwords in english I can think of
manual_stopwords = ['about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
manual_words = list(filter(lambda x: not( x.lower() in manual_stopwords), words))

# NLTK approach
stop_words = set(stopwords.words('english')) # This is a MASSIVE list of stopwords!!!
print("Some stopwords from NLTK:")
print(stop_words)
print()
# Remove stopwords manually:
nltk_words = list(filter(lambda x: not( x.lower() in stop_words), words))

## Results:
print("Original sentence:", sentence)
print("My filter:", " ".join(manual_words))
print("NLTK filter:", " ".join(nltk_words))