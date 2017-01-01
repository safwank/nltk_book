import nltk, re, pprint
import feedparser
import unicodedata
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import brown, gutenberg, nps_chat, udhr

stopwords = nltk.corpus.stopwords.words('english')

def process_html_content():
  url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
  html = request.urlopen(url).read().decode('utf8')
  raw = BeautifulSoup(html).get_text()
  tokens = word_tokenize(raw)
  words = normalize(tokens)
  return nltk.Text(words)

def process_rss_feed():
  llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
  post = llog.entries[2]
  content = post.content[0].value
  raw = BeautifulSoup(content).get_text()
  tokens = word_tokenize(raw)
  words = normalize(tokens)
  return nltk.Text(words)

def normalize(tokens):
  return [w.lower() for w in tokens if w.isalpha() and w.lower() not in stopwords]

def decode_and_encode_unicode():
  path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
  f = open(path, encoding='latin2')

  for line in f:
    line = line.strip()
    print(line)

def decode_and_encode_unicode_escaped():
  path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
  f = open(path, encoding='latin2')

  for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

def inspect_unicode_characters():
  path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
  lines = open(path, encoding='latin2').readlines()
  line = lines[2]
  print(line.encode('unicode_escape'))

  for c in line:
    if ord(c) > 127:
      print('{} U+{:04x} {}'.format(c, ord(c), unicodedata.name(c)))

def use_unicode_with_regex():
  path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
  lines = open(path, encoding='latin2').readlines()
  line = lines[2]
  line = line.lower()
  m = re.search('\u015b\w*', line)
  return m.group()

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

def regex_endswith():
  return [w for w in wordlist if re.search('ed$', w)]

def regex_wildcard_with_fixed_length():
  return [w for w in wordlist if re.search('^..j..t..$', w)]

def regex_optional(text=brown.words()):
  return sum(w for w in text if re.search('^e-?mail$', w))

def regex_textonyms():
  return [w for w in wordlist if re.search('^[ghi][mno][jkl][def]$', w)]

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))

def regex_closures1():
  return [w for w in chat_words if re.search('^m+i+n+e+$', w)]

def regex_closures2():
  return [w for w in chat_words if re.search('^[ha]+$', w)]

def regex_non_vowels():
  return [w for w in chat_words if re.search('^[^aeiouAEIOU]+$', w)]

wsj = sorted(set(nltk.corpus.treebank.words()))

def regex_escape1():
  return [w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]

def regex_escape2():
  return [w for w in wsj if re.search('^[A-Z]+\$$', w)]

def regex_occurences1():
  return [w for w in wsj if re.search('^[0-9]{4}$', w)]

def regex_occurences2():
  return [w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]

def regex_occurences3():
  return [w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]

def regex_endswith_or_sequence():
  return [w for w in wsj if re.search('(ed|ing)$', w)]

def regex_extract_word_pieces():
  word = 'supercalifragilisticexpialidocious'
  return re.findall(r'[aeiou]', word)

def regex_vowel_sequence_frequency():
  fd = nltk.FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word))
  return fd.most_common(12)

def regex_compartmentalize_date():
  return [int(n) for n in re.findall(r'[0-9]{1,4}', '2009-12-31')]

def regex_compress():
  def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)

  english_udhr = nltk.corpus.udhr.words('English-Latin1')
  regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
  return nltk.tokenwrap(compress(w) for w in english_udhr[:75])

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')

def regex_consonant_vowel_pair_cfd():
  cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
  cfd = nltk.ConditionalFreqDist(cvs)
  cfd.tabulate()

def regex_indexed_consonant_vowel_pairs(cv):
  cv_word_pairs = [(cv, w) for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
  cv_index = nltk.Index(cv_word_pairs)
  return cv_index[cv]

def regex_stemming(word):
  regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
  stem, suffix = re.findall(regexp, word)[0]
  return stem

def regex_search_tokenized_text1():
  moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
  return moby.findall(r'<a> <.*> <man>')

def regex_search_tokenized_text2():
  moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
  return moby.findall(r'<a> (<.*>) <man>')

chat = nltk.Text(nps_chat.words())

def regex_search_tokenized_text3():
  return chat.findall(r'<.*> <.*> <bro>')

def regex_search_tokenized_text4():
  return chat.findall(r'<l.*>{3,}')

hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))

def regex_discover_hypernyms():
  return hobbies_learned.findall(r'<\w*> <and> <other> <\w*s>')

def regex_discover_entity_properties():
  return hobbies_learned.findall(r'<as> <\w*> <as> <a> <\w*>')

stemming_raw = """DENNIS: Listen, strange women lying in ponds distributing swords
  is no basis for a system of government.  Supreme executive power derives from
  a mandate from the masses, not from some farcical aquatic ceremony."""
stemming_tokens = word_tokenize(stemming_raw)

def stem_porter():
  porter = nltk.PorterStemmer()
  return [porter.stem(t) for t in stemming_tokens]

def stem_lancaster():
  lancaster = nltk.LancasterStemmer()
  return [lancaster.stem(t) for t in stemming_tokens]

def lemmatization():
  wnl = nltk.WordNetLemmatizer()
  return [wnl.lemmatize(t) for t in stemming_tokens]

tokenization_raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
  though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
  well without--Maybe it's always pepper that makes people hot-tempered,'..."""

def tokenize_with_whitespace():
  return re.split(r' ', tokenization_raw)

def tokenize_with_regex_space_tab_newline():
  return re.split(r'[ \t\n]+', tokenization_raw)

def tokenize_with_regex_any_whitespace():
  return re.split(r'\s+', tokenization_raw)

def tokenize_with_regex_words():
  return re.findall(r'\w+', tokenization_raw)

def tokenize_with_regex_non_words():
  return re.split(r'\W+', tokenization_raw)

def tokenize_with_regex_ultimate():
  return re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", tokenization_raw)

# NOTE: this doesn't work as expected
def tokenize_with_nltk_regex_tokenizer():
  text = 'That U.S.A. poster-print costs $12.40...'
  pattern = r'''(?x)    # set flag to allow verbose regexps
      ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes (]), (])
  '''
  return nltk.regexp_tokenize(text, pattern)

def segment_text_with_sentence_tokenizer():
  text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
  sents = nltk.sent_tokenize(text)
  pprint.pprint(sents[79:89])

def segment(text, segs):
  words = []
  last = 0
  for i in range(len(segs)):
    if segs[i] == '1':
      words.append(text[last:i+1])
      last = i+1
  words.append(text[last:])
  return words

def segmentation_example():
  text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
  seg1 = "0000000000000001000000000010000000000000000100000000000"
  seg2 = "0100100100100001001001000010100100010010000100010010000"
  print(segment(text, seg1))
  print(segment(text, seg2))

# Objective function that calculates the score of the quality of the
# segmentation (smaller is better).
def evaluate_segmentation(text, segs):
  words = segment(text, segs)
  text_size = len(words)
  lexicon_size = sum(len(word) + 1 for word in set(words))
  return text_size + lexicon_size

def evaluate_segmentation_example():
  text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
  seg1 = "0000000000000001000000000010000000000000000100000000000"
  seg2 = "0100100100100001001001000010100100010010000100010010000"
  seg3 = "0000100100000011001000000110000100010000001100010000001"
  print(evaluate_segmentation(text, seg3))
  print(evaluate_segmentation(text, seg2))
  print(evaluate_segmentation(text, seg1))

# Optimization using simulated annealing to find the best segmentation
from random import randint

def flip(segs, pos):
  return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
  for i in range(n):
    segs = flip(segs, randint(0, len(segs)-1))
  return segs

def anneal(text, segs, iterations, cooling_rate):
  temperature = float(len(segs))
  while temperature > 0.5:
    best_segs, best = segs, evaluate_segmentation(text, segs)
    for i in range(iterations):
      guess = flip_n(segs, round(temperature))
      score = evaluate_segmentation(text, guess)
      if score < best:
        best, best_segs = score, guess
    score, segs = best, best_segs
    temperature = temperature / cooling_rate
    print(evaluate_segmentation(text, segs), segment(text, segs))
  print()
  return segs

def anneal_example():
  text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
  seg1 = "0000000000000001000000000010000000000000000100000000000"
  anneal(text, seg1, 5000, 1.2)

# String formatting example - tabulating a CFD
def tabulate(cfdist, words, categories):
  print('{:16}'.format('Category'), end=' ')                  # column headings
  for word in words:
    print('{:>6}'.format(word), end=' ')
  print()
  for category in categories:
    print('{:16}'.format(category), end=' ')                  # row heading
    for word in words:                                        # for each word
      print('{:6}'.format(cfdist[category][word]), end=' ')   # print table cell
    print()                                                   # end the row

def tabulate_example():
  cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
  genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
  modals = ['can', 'could', 'may', 'might', 'must', 'will']
  tabulate(cfd, modals, genres)

# Text wrapping example
from textwrap import fill

def wrap_text():
  saying = ['After', 'all', 'is', 'said', 'and', 'done', ',',
    'more', 'is', 'said', 'than', 'done', '.']
  format = '%s (%d),'
  pieces = [format % (word, len(word)) for word in saying]
  output = ' '.join(pieces)
  wrapped = fill(output)
  print(wrapped)

def reverse_text(text='monty-python'):
  return text[::-1]

def calculate_readability_index(words=brown.words(), sents=brown.sents()):
  char_count = sum(len(w) for w in words)
  word_count = len(words)
  sent_count = len(sents)
  avg_letters_per_word = char_count / word_count
  avg_words_per_sent = word_count / sent_count
  return (4.71 * avg_words_per_sent) + (0.5 * avg_letters_per_word) - 21.43

def calculate_readability_index_example():
  return sorted((calculate_readability_index(brown.words(categories=genre), brown.sents(categories=genre)), genre)
    for genre in brown.categories())

# https://en.wikipedia.org/wiki/Soundex#American_Soundex
def soundex(word):
  mapping = {
    '[bfpvBFPV]': '1',
    '[cgjkqsxzCGJKQSXZ]': '2',
    '[dtDT]': '3',
    '[lL]': '4',
    '[mnMN]': '5',
    '[rR]': '6'
  }
  rest = word[0] + re.sub(r'[hw]', '', word[1:])

  for pattern, digit in mapping.items():
    rest = re.sub(pattern, digit, rest)

  rest = re.sub(r'(.)\1+', r'\1', rest)
  rest = rest[0] + re.sub(r'[aeiouy]', '', rest[1:])

  if (rest[0].isdigit()):
    rest = word[0] + rest[1:]

  return rest[:4].ljust(4, '0')

# Guess language of previously unseen text
def bigram_freqdist(words):
  return nltk.FreqDist("".join(w)
    for word in words
    for w in nltk.bigrams(word.lower()))

en_fd = bigram_freqdist(udhr.words("English-Latin1"))
fr_fd = bigram_freqdist(udhr.words("French_Francais-Latin1"))
de_fd = bigram_freqdist(udhr.words("German_Deutsch-Latin1"))
es_fd = bigram_freqdist(udhr.words("Spanish-Latin1"))
langs = ['en', 'fr', 'de', 'es']

def label_ranks(ranks):
  for i in range(len(langs)):
    yield ranks[i], langs[i]

def guess_language(words):
  ranks = list(map(lambda x: nltk.spearman_correlation(x, bigram_freqdist(words)),
    [en_fd, fr_fd, de_fd, es_fd]))
  print(ranks)
  return sorted(label_ranks(ranks), reverse=True)[0][1]
