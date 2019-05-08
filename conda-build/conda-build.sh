#!/usr/bin/env bash

#conda config --set anaconda_upload yes
conda-build . -c pytorch # use pytorch branch to install pytorch