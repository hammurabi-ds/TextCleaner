from nltk.corpus import stopwords
import re
from src import full_stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from polyglot.tag import POSTagger as PG_POSTagger
from nltk import pos_tag as nltk_pos_tag
from lemmy import lemmatizer
from nltk.stem import WordNetLemmatizer as WNL
from nltk.corpus import wordnet
from string import punctuation
from sklearn.base import TransformerMixin
import pandas as pd
from polyglot.detect import Detector


class TextCleaner():

    def __init__(self, lang="english"):
        """
        """
        self.lang = lang
        self.punctuation = list(punctuation)

    def sentence_tokenize(self, text_list):
        """
        tokenize a list of documents into sentences.

        :param text_list: raw text corpus
        :rtype: Tokenized sentences
        """
        return [sent_tokenize(text, language=self.lang) for text in text_list]

    def word_tokenize(self, text_list):
        """
        tokenize a list of documents into words .

        :param text_list: raw text corpus
        :rtype: Tokenized words
        """
        return [word_tokenize(text, language=self.lang) for text in text_list]

    def stem(self, word_list):
        """
        stem words into root forms.

        :param text_list: raw text corpus
        :rtype: stemmed words
        """
        sno = SnowballStemmer(self.lang)
        return [sno.stem(word) for word in word_list]

    def pos_tag(self, word_list):
        """
        tag word by part-of-speech using polyglot.

        :param text_list: tokenized text corpus
        :rtype: POS tagged words
        """
        if (self.lang.lower() == "danish"):
            tagger = PG_POSTagger(lang="da")
            return list(tagger.annotate(word_list))
        elif (self.lang.lower() == "swedish"):
            tagger = PG_POSTagger(lang="sv")
            return list(tagger.annotate(word_list))
        elif (self.lang.lower() == "english"):
            tagger = PG_POSTagger(lang="en")
            return list(tagger.annotate(word_list))
        else:
            pass

    def lemmatize(self, tag_list):
        """
        lemmatize word into canonical forms .

        :param tag_list: tokenized word list
        :rtype: Lemmantized words
        """
        if (self.lang.lower() == "danish"):
            lem = lemmatizer.load("da")
            return [lem.lemmatize(tag, word)[0] for word, tag in tag_list]
        elif (self.lang.lower() == "swedish"):
            lem = lemmatizer.load("sv")
            return [lem.lemmatize(tag, word)[0] for word, tag in tag_list]
        elif (self.lang.lower() == "english"):
            wnl = WNL()
            return [wnl.lemmatize(word, self._get_wordnet_pos(tag)) for word, tag in tag_list]

    def tag_and_lemmatize(self, word_list):
        """
        tag and lemmatize.

        :param text_list: tokenized words list
        :rtype: Lemmantized and tagged words using nltk
        """
        return self.lemmatize(self.pos_tag(word_list))

    def get_words_by_tag(self, tag_list, tags):
        """
        get words according to a tag list of desired tags.

        :param tag_list: A list of desired tags.
        example ["VB","CD","NN"] which keeps verbs, digits and nouns.
        :param tags: tagged words
        :rtype: tagged words in tag_list.
        """
        return [(word, tag) for (word, tag) in tag_list if tag in tags]

    def remove_stopwords(self, word_list, full=False):
        """
        remove common stopwords. user may specify whether
        to use the full word list or not.

        :param word_list: a list of words contained in document
        :param full: if True use the full danish stoplist
        provided by the package, else nltk stopwords
        :rtype: Words not in stoplist
        """
        if full:
            sw = full_stopwords.FULL_STOPWORDS[self.lang]
        else:
            sw = stopwords.words(self.lang)
        return [word for word in word_list if word not in sw]

    def remove_punctuation(self, word_list, punctuation=None):
        """
        remove punctuations. user may specify punctuation set.
        default set is string.punctuation: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~.

        :param word_list: a list of words contained in document
        :rtype: word list without punctuation
        """
        if punctuation is None:
            punctuation = self.punctuation
        return [word for word in word_list if word not in punctuation]

    def remove_numbers(self, word_list):
        """
        remove any numbers in each word.

        :param word_list: a list of words contained in document
        :rtype: word list without numbers
        """
        return [re.sub(r'\d+', '', word) for word in word_list]

    def remove_symbols(self, word_list, symbols):
        """
        remove user-specified symbols.

        :param word_list: a list of words contained in document
        :param symbols: a list of symbols you need removed
        :rtype: word list without symbols you specified in list
        """
        return [word for word in word_list if word not in symbols]

    def remove_empty_words(self, word_list):
        """
        remove empty words, i.e. ''

        :param word_list: a list of words contained in document
        :rtype: a list of non-empty words
        """
        return [word for word in word_list if not word == ""]

    def regex_sub(self, pattern_replace, doc_list):
        """
        apply sequential regex substitutes for each document. 
        use as the initial step in pipeline

        :param pattern_replace: (list of) tuple of str: (pattern, replace)
        :param doc_list: list of document strings, [ doc ].
        :rtype: a document with regex subs
        """
        return [self._regex_sub_doc(pattern_replace, doc) for doc in doc_list]

    def _regex_sub_doc(self, pattern_replace, doc):
        if isinstance(pattern_replace, tuple):
            pattern_replace = [pattern_replace]
        for p, r in pattern_replace:
            doc = re.sub(p, r, doc)
        return doc
        
    def _get_wordnet_pos(self, tag):
        """
        get POStag in wordnet format
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

class UnitTransformer():
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X


class Stemmer(UnitTransformer):
    def transform(self, word_list,):
        """
        input: list of list of words
        """
        return [self.preprocessor.stem(wl) for wl in word_list]


class WordTokenizer(UnitTransformer):
    def transform(self, text_list):
        return self.preprocessor.word_tokenize(text_list)


class SentenceTokenizer(UnitTransformer):
    def transform(self, text_list):
        return self.preprocessor.sentence_tokenize(text_list)


class POSTagger(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.pos_tag(wl) for wl in word_list]


class Lemmatizer(UnitTransformer):
    def transform(self, tag_list):
        """
        input: list of list of (word, tag) tuples
        """
        return [self.preprocessor.lemmatize(tl) for tl in tag_list]


class WordByTag(UnitTransformer):
    def __init__(self, prep, tags):
        self.preprocessor = prep
        self.tags = tags

    def transform(self, tag_list):
        return [self.preprocessor.get_words_by_tag(tl, self.tags) for tl in tag_list]


class PunctuationCleaner(UnitTransformer):

    def __init__(self, preprocessor, punctuation=None):
        UnitTransformer.__init__(self, preprocessor)
        self.punctuation = punctuation

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_punctuation(wl, self.punctuation) for wl in word_list]


class StopwordCleaner(UnitTransformer):
    def __init__(self, preprocessor, full=False):
        UnitTransformer.__init__(self, preprocessor)
        self.full = full

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_stopwords(wl, self.full) for wl in word_list]


class NumberCleaner(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_numbers(wl) for wl in word_list]


class SymbolCleaner(UnitTransformer):
    def __init__(self, preprocessor, symbols):
        UnitTransformer.__init__(self, preprocessor)
        self.symbols = symbols

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_symbols(wl, self.symbols) for wl in word_list]


class EmptywordCleaner(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_empty_words(wl) for wl in word_list]


class RegexSub(UnitTransformer):
    def __init__(self, preprocessor, pattern_replace):
        self.preprocessor = preprocessor
        self.pattern_replace = pattern_replace

    def transform(self, doc_list):
        return self.preprocessor.regex_sub(self.pattern_replace, doc_list)






