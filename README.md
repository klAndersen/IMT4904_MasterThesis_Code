# IMT4904 MasterThesis Source code

Python repository for my Master thesis @ NTNU Gjøvik.
Based on Scikit-learn (developer version v18.0: [http://scikit-learn.org/dev/]()) 
and dataset from StackExchange: StackOverflow (available here: [https://archive.org/details/stackexchange]()).

Quick requirement list:
- Python 3.5
- Numpy
- Scipy
- Pandas
- Cython
- Openblas and lapack
- BeatifulSoup4 (bs4) 
- mysql-connector-python: [https://github.com/mysql/mysql-connector-python]()
- Natural Language Toolkit (nltk); use *nltk.download()* and aquire the resource *'tokenizers/punkt/english.pickle'*.
- Scikit-learn >= dev.v18 (available from GitHub: [https://github.com/scikit-learn/scikit-learn]())
- StackOverflow dataset added to MySQL (not included in project)
- Before starting the program, ensure that you have a dummy file named *"dbconfig.py"* (just rename dbconfig.py.example)
