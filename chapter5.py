import nltk
from nltk.corpus import brown, treebank

brown_text = nltk.Text(word.lower() for word in brown.words())

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
