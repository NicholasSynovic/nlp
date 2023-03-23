#!/bin/bash

wget -O corpus.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip?ref=blog.salesforceairesearch.com
unzip corpus.zip
rm corpus.zip

wget -O corpus.tar.gz http://alfonseca.org/pubs/ws353simrel.tar.gz
tar -xzf corpus.tar.gz
rm corpus.tar.gz
