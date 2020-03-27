#!/bin/bash

ROOT=$PWD

mkdir tmp && cd tmp &&\
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz && \
tar xvf rt-polaritydata.tar.gz && \
mv rt-polaritydata "$ROOT/data" && \
cd $ROOT && \
rm -r tmp && \
python src/preprocess.py data/rt-polarity.pos data/rt-polarity.neg pkl