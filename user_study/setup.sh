#!/bin/bash

# Configure Eval server
echo 'Setting up Eval server..'
python3.9 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
pip install -r requirements.txt
