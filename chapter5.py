import nltk
import pylab
from nltk.corpus import brown, treebank

brown_text = nltk.Text(word.lower() for word in brown.words())
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

def tag(text):
  tokens = nltk.word_tokenize(text)
  return nltk.pos_tag(tokens)

def pos_tagging1():
  return tag("And now for something completely different")

def pos_tagging2():
  return tag("They refuse to permit us to obtain the refuse permit")

def find_similar(word, text=brown_text):
  return text.similar(word)

def get_tagged_words_in_corpus(corpus=brown):
  return corpus.tagged_words()

def get_universal_tagged_words_in_corpus(corpus=brown):
  return corpus.tagged_words(tagset='universal')

def get_tagged_sentences(corpus=brown):
  return corpus.tagged_sents()

def get_tag_fd(corpus=brown):
  tagged = corpus.tagged_words(tagset='universal')
  tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
  return tag_fd.most_common()

def get_most_common_noun_preceders(corpus=brown):
  tagged = corpus.tagged_words(tagset='universal')
  word_tag_pairs = nltk.bigrams(tagged)
  noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
  fdist = nltk.FreqDist(noun_preceders)
  return [tag for (tag, _) in fdist.most_common()]

def get_most_common_verbs(corpus=treebank):
  tagged = corpus.tagged_words(tagset='universal')
  word_tag_fd = nltk.FreqDist(tagged)
  return [wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB']

def get_word_tag_cfd(word, corpus=treebank):
  tagged = corpus.tagged_words(tagset='universal')
  cfd = nltk.ConditionalFreqDist(tagged)
  return cfd[word].most_common()

def get_past_and_past_participle_verbs(corpus=treebank):
  tagged = corpus.tagged_words()
  cfd = nltk.ConditionalFreqDist(tagged)
  return [w for w in cfd.conditions() if 'VBD' in cfd[w] and 'VBN' in cfd[w]]

def get_most_common_past_participle_preceders(corpus=treebank):
  tagged = corpus.tagged_words()
  word_tag_pairs = nltk.bigrams(tagged)
  preceders = [a for (a, b) in word_tag_pairs if b[1] == 'VBN']
  fdist = nltk.FreqDist(preceders)
  return [tag for (tag, _) in fdist.most_common()]

def find_tags(tag_prefix, tagged_text):
  cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
  return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())

def find_most_common_noun_tags():
  tags = find_tags('NN', nltk.corpus.brown.tagged_words(categories='news'))
  for tag in sorted(tags):
    print(tag, tags[tag])

def find_words_that_follow(word='often'):
  brown_learned_text = brown.words(categories='learned')
  return sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))

def tabulate_tags_that_follow(word='often'):
  brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
  tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
  fd = nltk.FreqDist(tags)
  fd.tabulate()

def find_verb_to_verb(corpus=brown):
  for sentence in brown.tagged_sents():
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
      if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
        yield w1, w2, w3

def print_ambiguous_words():
  brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
  data = nltk.ConditionalFreqDist((word.lower(), tag)
                                  for (word, tag) in brown_news_tagged)
  for word in sorted(data.conditions()):
    if len(data[word]) > 3:
      tags = [tag for (tag, _) in data[word].most_common()]
      print(word, ' '.join(tags))

def lookup_anagrams(word):
  words = nltk.corpus.words.words('en')
  anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
  return anagrams[word]

def default_tagging():
  tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
  most_likely_tag = nltk.FreqDist(tags).max()

  raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
  tokens = nltk.word_tokenize(raw)
  default_tagger = nltk.DefaultTagger('NN')

  print(default_tagger.evaluate(brown_tagged_sents))

def regex_tagging():
  patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                     # nouns (default)
  ]
  regexp_tagger = nltk.RegexpTagger(patterns)
  print(regexp_tagger.evaluate(brown_tagged_sents))

def lookup_tagging(n=100):
  fd = nltk.FreqDist(brown.words(categories='news'))
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))

  most_freq_words = fd.most_common(n)
  likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)

  baseline_tagger = nltk.UnigramTagger(model=likely_tags)
  print(baseline_tagger.evaluate(brown_tagged_sents))

def tagging_performance(cfd, wordlist):
  lt = dict((word, cfd[word].max()) for word in wordlist)
  baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
  return baseline_tagger.evaluate(brown_tagged_sents)

def plot_tagging_performance():
  word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
  words_by_freq = [w for (w, _) in word_freqs]
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
  sizes = 2 ** pylab.arange(15)
  perfs = [tagging_performance(cfd, words_by_freq[:size]) for size in sizes]
  pylab.plot(sizes, perfs, '-bo')
  pylab.title('Lookup Tagger Performance with Varying Model Size')
  pylab.xlabel('Model Size')
  pylab.ylabel('Performance')
  pylab.show()

def trained_lookup_tagging():
  unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
  print(unigram_tagger.evaluate(brown_tagged_sents))

def split_dataset():
  size = int(len(brown_tagged_sents) * 0.9)
  train_sents = brown_tagged_sents[:size]
  test_sents = brown_tagged_sents[size:]
  return train_sents, test_sents

def properly_trained_lookup_tagging():
  train_sents, test_sents = split_dataset()
  unigram_tagger = nltk.UnigramTagger(train_sents)
  print(unigram_tagger.evaluate(test_sents))

def trained_bigram_tagging():
  train_sents, test_sents = split_dataset()
  bigram_tagger = nltk.BigramTagger(train_sents)
  print(bigram_tagger.evaluate(test_sents))

def combined_tagging():
  train_sents, test_sents = split_dataset()
  t0 = nltk.DefaultTagger('NN')
  t1 = nltk.UnigramTagger(train_sents, backoff=t0)
  t2 = nltk.BigramTagger(train_sents, backoff=t1)
  t3 = nltk.TrigramTagger(train_sents, backoff=t2)
  print(t3.evaluate(test_sents))

def calculate_tagging_ambiguity():
  cfd = nltk.ConditionalFreqDist(
             ((x[1], y[1], z[0]), z[1])
             for sent in brown_tagged_sents
             for x, y, z in nltk.trigrams(sent))
  ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
  return sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()

def tagging_confusion_matrix():
  train_sents, test_sents = split_dataset()
  t0 = nltk.DefaultTagger('NN')
  t1 = nltk.UnigramTagger(train_sents, backoff=t0)
  t2 = nltk.BigramTagger(train_sents, backoff=t1)

  test_tags = [tag for sent in brown.sents(categories='editorial')
                   for (word, tag) in t2.tag(sent)]
  gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
  print(nltk.ConfusionMatrix(gold_tags, test_tags))

def get_unique_tags():
  return sorted(set([tag for sent in brown_tagged_sents for word, tag in sent]))

def words_with_most_distinct_tags(n=10):
  cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
  top_words = sorted([(len(cfd[word]), word) for word in cfd.conditions()], reverse=True)[:n]
  return [(word, tag_count) for tag_count, word in top_words]

def most_common_tags(n=20):
  all_tags = [tag for sent in brown_tagged_sents for word, tag in sent]
  fd = nltk.FreqDist(all_tags)
  return fd.most_common(n)
