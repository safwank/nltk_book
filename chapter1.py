import nltk
from nltk.book import *

# word types / tokens
def lexical_diversity(text):
  return len(set(text)) / len(text)

def percentage(count, total):
  return 100 * count / total
