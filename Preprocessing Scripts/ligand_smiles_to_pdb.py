# Converts ligand SMILES strings into PDB files
# reads from smiles.csv and outputs to the ligand_pdb directory.

import csv
import mechanize
import os
from pyquery import PyQuery
import ssl
import urllib.request

HTML_SUCCESS = 200
URL_TEMPLATE = "http://cactus.nci.nih.gov{}"
FILE_TEMPLATE = "{}/{}.pdb"
URL = URL_TEMPLATE.format('/translate')
OUTDIR = 'ligand_pdb'
DATA_FILE = 'data.txt'

ssl._create_default_https_context = ssl._create_unverified_context
br = mechanize.Browser()
br.set_handle_robots(False)

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

with open(DATA_FILE, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    rows = list(reader)[1:]
    data = set(( (row[1], row[2]) for row in rows ))
    progress = 0
    for common, smiles in data:
        progress += 1
        print("Converting compound {}/{}: {}".format(progress, len(data), common))
        print('  Opening connection')
        br.open(URL)
        br.select_form(name='form')
        br['smiles'] = smiles
        br['format'] = ['pdb']
        br['astyle'] = ['aromatic']
        br['dim'] = ['3D']
        print('  Submitting form')
        res = br.submit()
        if res.getcode() == HTML_SUCCESS:
            print('  Downloading PDB file')
            content = res.read()
            pq = PyQuery(content)
            link = pq('a:first').attr('href')
            file_url = URL_TEMPLATE.format(link)
            filename = FILE_TEMPLATE.format(OUTDIR, common)
            urllib.request.urlretrieve(file_url, filename)
            print("  Success! File saved to {}".format(filename))
        else:
            print("  Invalid form response: {}".format(res.getcode()))
        print()
