# IMT4904 MasterThesis: Test all algorithms 

This folder contains the pickled models that was created by running test_all_algorithms.py.

It should be noted that these were created with only has_codeblock, has_numeric and has_hexadecimal feature detection.
In addition, only those with 'cv' was trained by using Cross-validation. 

The following is a short description of the models:
- 's_': Model trained by use of stemming
- 'u_': Model trained by using unprocessed data set
- 'cv': Model trained by using cross-validation
- 'tfid': Model was trained by using TfidfVectorizer (the rest used CountVectorizer)
- '\*\_detector\_type': Type indicates the algorithm used to train the model
- '_stopwords': Model trained by having *stop_words='english'*