#!/usr/bin/python
# -*- coding: utf-8 -*
##############################################################
# File Name: extract.py
# Author:
# mail:
# Created Time: Sun Mar  2 12:26:24 2025
# Brief:
##############################################################

import sys
import json

def extract_code_from_ipynb(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    code = ""
    for cell in code_cells:
        code += "".join(cell['source']) + "\n"

    return code

#
if len(sys.argv) < 2:
    print("Usage: python extract.py <arg1:input ipynb file> <arg2:output py file> ...")
    sys.exit(1)

extracted_code = extract_code_from_ipynb(sys.argv[1])
print(extracted_code)
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(extracted_code);
