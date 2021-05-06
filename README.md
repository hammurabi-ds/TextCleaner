# TextCleaner
Text cleaning tools usable in sklearn pipelines


```python
    prep = TextCleaner("english")
    
    symbols = ['¤¤, ##, &&'']
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
    
    # Fit the pipeline to data
    clean_text = steps.fit_transform(text)
    
    # Merge the cleaned words into sentences again
    data=[' '.join(x) for x in clean_text]
    
```

Which returns the following

```
['email address mail outlook.com',
 'hello email address mail gmail.com',
 'credit card number CREDIT CARD']
```

