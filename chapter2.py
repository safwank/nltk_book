import nltk
import string
from nltk.corpus import gutenberg, webtext, brown, inaugural, udhr, swadesh, wordnet

def get_gutenberg_statistics():
  for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)

def get_webtext_raw():
  for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

def get_brown_stylistics():
  news_text = brown.words(categories='news')
  fdist = nltk.FreqDist(w.lower() for w in news_text)
  modals = ['can', 'could', 'may', 'might', 'must', 'will']

  for m in modals:
    print(m + ':', fdist[m], end=' ')

def get_brown_stylistics_cfd():
  cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
  genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
  modals = ['can', 'could', 'may', 'might', 'must', 'will']
  cfd.tabulate(conditions=genres, samples=modals)

def lexical_diversity(text):
  return len(set(text)) / len(text)

def tabulate_genre_lexical_diversity():
  for genre in brown.categories():
    yield (lexical_diversity(brown.words(categories=genre)), genre)

def get_american_citizens_cfd():
  cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
  cfd.plot()

def get_udhr_word_length_cdf():
  languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut',
    'Hungarian_Magyar', 'Malay_BahasaMelayu']
  cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
  cfd.plot(cumulative=True)

def get_newsworthy_vs_romantic_vs_religious_days():
  cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
  genres = ['news', 'romance', 'religion']
  days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  cfd.tabulate(conditions=genres, samples=days)

# Obtains all bigrams from a text and constructs
# a conditional frequency distribution to record
# which words are most likely to follow a given word;
# e.g., after the word living, the most likely word
# is creature; the generate_random_text() function
# uses this data, and a seed word, to generate random text.
def generate_random_text(text, word, num=15):
  bigrams = nltk.bigrams(text)
  cfd = nltk.ConditionalFreqDist(bigrams)

  for i in range(num):
    print(word, end=' ')
    word = cfd[word].max()

def most_frequent_pairs_of_words(text = brown.words(), max=50):
  bigrams = nltk.bigrams(text)
  stopwords = nltk.corpus.stopwords.words('english')
  fd = nltk.FreqDist(pair for pair in bigrams
    if pair[0] not in stopwords and pair[1] not in stopwords and
    pair[0] not in string.punctuation and pair[1] not in string.punctuation)
  return fd.most_common()[:max]

def unusual_words(text):
  text_vocab = set(w.lower() for w in text if w.isalpha())
  english_vocab = set(w.lower() for w in nltk.corpus.words.words())
  unusual = text_vocab - english_vocab
  return sorted(unusual)

def content_fraction(text):
  stopwords = nltk.corpus.stopwords.words('english')
  content = [w for w in text if w.lower() not in stopwords]
  return len(content) / len(text)

def generate_target_puzzle():
  puzzle_letters = nltk.FreqDist('egivrvonl')
  obligatory = 'r'
  wordlist = nltk.corpus.words.words()
  return [w for w in wordlist if len(w) >= 4
    and obligatory in w
    and nltk.FreqDist(w) <= puzzle_letters]

def prove_zipfs_law(text=brown.words(), max=30):
  words = [w.lower() for w in text]
  fd = nltk.FreqDist(words)
  return fd.most_common(max)

def get_ambiguous_names():
  names = nltk.corpus.names
  male_names = names.words('male.txt')
  female_names = names.words('female.txt')
  return [w for w in male_names if w in female_names]

def get_last_letter_of_name_cfd():
  names = nltk.corpus.names
  cfd = nltk.ConditionalFreqDist(
    (fileid, name[-1])
    for fileid in names.fileids()
    for name in names.words(fileid))
  cfd.plot()

def get_words_with_3_phones():
  entries = nltk.corpus.cmudict.entries()

  for word, pron in entries:
    if len(pron) == 3:
      ph1, ph2, ph3 = pron
      if ph1 == 'P' and ph3 == 'T':
        print(word, ph2, end=' ')

def find_rhyming_words():
  entries = nltk.corpus.cmudict.entries()
  syllable = ['N', 'IH0', 'K', 'S']
  return [word for word, pron in entries if pron[-4:] == syllable]

def get_pronunciation_mismatches1():
  entries = nltk.corpus.cmudict.entries()
  return [w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']

def get_pronunciation_mismatches2():
  entries = nltk.corpus.cmudict.entries()
  return sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))

def stress(pron):
  return [char for phone in pron for char in phone if char.isdigit()]

def find_stress_pattern1():
  entries = nltk.corpus.cmudict.entries()
  return [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]

def find_stress_pattern2():
  entries = nltk.corpus.cmudict.entries()
  return [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]

def find_minimally_contrasting_words():
  entries = nltk.corpus.cmudict.entries()
  p3 = [(pron[0]+'-'+pron[2], word)
    for (word, pron) in entries
    if pron[0] == 'P' and len(pron) == 3]
  cfd = nltk.ConditionalFreqDist(p3)

  for template in sorted(cfd.conditions()):
    if len(cfd[template]) > 10:
      words = sorted(cfd[template])
      wordstring = ' '.join(words)
      print(template, wordstring[:70] + "...")

def words_with_longest_syllables(max=10):
  entries = nltk.corpus.cmudict.entries()
  syllables = [(len(pron), word) for (word, pron) in entries]
  syllables.sort(reverse=True)
  return syllables[:max]

def compare_germanic_and_latin_words():
  languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'it', 'la']
  for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])

# synset -> synonym set (sense)
# lemmas -> collection of synonymous words in a synset
def get_synonyms(word):
  for synset in wordnet.synsets(word):
    yield synset.lemma_names()

# hyponym -> specific (top to bottom)
def get_types_of_car():
  motorcar = wordnet.synset('car.n.01')
  types_of_motorcar = motorcar.hyponyms()
  return sorted(lemma.name() for synset in types_of_motorcar
    for lemma in synset.lemmas())

# hypernym -> generalization (bottom up)
def navigate_car_hierarchy():
  motorcar = wordnet.synset('car.n.01')
  return [synset.name() for path in motorcar.hypernym_paths() for synset in path]

def print_tree_parts():
  tree = wordnet.synset('tree.n.01')
  print(tree.part_meronyms())
  print(tree.substance_meronyms())
  print(tree.member_holonyms())

def get_most_similar_words(synset1, synset2):
  return synset1.lowest_common_hypernyms(synset2)

def calculate_similarity_score(synset1, synset2):
  return synset1.path_similarity(synset2)

def words_with_highest_polysemy(max=10):
  polysemy = [(len(wordnet.synsets(word)), word) for word in wordnet.words()]
  polysemy.sort(reverse=True)
  return polysemy[:max]
