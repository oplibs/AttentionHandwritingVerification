#!/usr/bin/python
# -*- coding: utf-8 -*
##############################################################
# File Name: 1.init_data.py
# Author:
# mail:
# Created Time: Sun Mar  2 12:48:53 2025
# Brief:
##############################################################
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ishanikathuria/handwritten-signature-datasets")

print("Path to dataset files:", path)
#
