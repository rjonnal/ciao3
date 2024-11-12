#!/usr/bin/bash
./cleanup.sh
python setup.py build_ext --inplace
rm -rfv build
#python test_cython_only.py
cp centroid.cpython-311-x86_64-linux-gnu.so ../centroid.so
