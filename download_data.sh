#!/usr/bin/env bash

wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz
wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Movies_and_TV.json.gz
gzip -d Movies_and_TV.json.gz
gzip -d meta_Movies_and_TV.json.gz
mv Movies_and_TV.json spark_workspace/
mv meta_Movies_and_TV.json spark_workspace/