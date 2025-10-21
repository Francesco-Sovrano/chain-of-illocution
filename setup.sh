#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Configure Eval server
cd user_study
echo 'Setting up the User Study server..'
python3.10 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
pip install -r requirements.txt
cd ..

cd explanation_analysis
python3.10 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
echo 'Installing KnowPy'
pip install -e $SCRIPT_PATH/knowpy
pip install -r requirements.txt

# python3 -m spacy download en_core_web_lg
# python3 -m spacy download de_dep_news_lg
# python3 -m spacy download fr_dep_news_lg
# python3 -m spacy download it_core_news_lg

python -m spacy download en_core_web_md
# python3 -m spacy download de_core_news_md
# python3 -m spacy download fr_core_news_md
# python3 -m spacy download it_core_news_md

python -m nltk.downloader stopwords punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng framenet_v17 wordnet brown omw-1.4
cd ..
