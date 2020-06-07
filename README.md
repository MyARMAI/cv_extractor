# myArmAi
![Docker](https://github.com/MyARMAI/cv_extractor/workflows/Docker/badge.svg?branch=master)
MyARM CV fields Extractor. There's two types of extractor :

1. Rule based extractor
2. Named Entity Recognition based extractor

The rule based extractor Word on Windows based system because of the usage of WIN32 api client
used for converting doc to docx file.

Dependencies :

- python-docx
- flask
- tensorflow (1.x)
- nltk
- keras
- gensim
- textract
- antiword if on WIN OS

# Getting Started

- cd into app/src
- python index.py
- Open your navigator and go to : http://127.0.0.1:5000/
- For testing : send a Resume (CV) in docx or doc format(pdf only work if using the ner extractor) post request to /v1/api/extractor/extract
