import pytest

from txtclnr.txtclnr import *
from sklearn.pipeline import Pipeline

__author__ = "hammurabi-ds"
__copyright__ = "hammurabi-ds"
__license__ = "MIT"


def test_pipeline():
    """API Tests"""

    prep = TextCleaner("english")
    symbols = ['##, &&']
    regex = ("\d{16}|\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}", 'CREDIT CARD')

    # Do basic preprocessing using a pipeline
    steps = Pipeline([        
        ('RE',RegexSub(prep, regex)),
        ('word_tokenize', WordTokenizer(prep)),
        ('number', NumberCleaner(prep)),
        ('punctuation', PunctuationCleaner(prep)),
        ('stopword', StopwordCleaner(prep, False)),
        ('SymbolRemover',SymbolCleaner(prep, symbols)),
        ('empty', EmptywordCleaner(prep)),
        ('pos', POSTagger(prep)),
        ('lemma', Lemmatizer(prep))
    ])
    
    text = ["&& my email address is : mail@outlook.com", 
            "hello ## my email address is, mail2@gmail.com", 
            "my credit card number is 3221 1111 1111 1111"]
    clean_text = steps.fit_transform(text)
    data=[' '.join(x) for x in clean_text]1

    assert data[2].find("CREDIT CARD")!=-1
    assert data[0].find("is")==-1
    assert len(data)==3

