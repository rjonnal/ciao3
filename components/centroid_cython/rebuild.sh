#!/usr/bin/bash
./cleanup.sh
python setup.py build_ext --inplace
rm -rfv build
cp centroid.cpython*.so ../centroid.so
