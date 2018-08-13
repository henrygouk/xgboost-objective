#!/bin/bash

mkdir -p xgboost/build
cd xgboost/build
cmake ..
make -j
cd ../../

PYTHONPATH=`pwd`/xgboost/python-package/ python runexp.py --dataset=higgs --objective=hinge --eta=1
