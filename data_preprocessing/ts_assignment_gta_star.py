# import pytest

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


import pytest

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from rdflib.exceptions import ParserError

from rdflib import Graph
from rdflib.util import guess_format


from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib.serializer import Serializer

import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF
from rdflib.namespace import FOAF

from rdflib import Graph
from rdflib import  term
# import rdflib.rdflib

import pytest

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import csv

# import rdflib
from datetime import datetime

print('use ntstarenv conda environment')

print('this is very slow and can take up to days')
# source of rdflib: https://github.com/XuguangSong98/rdflib

g = Graph()

name = 'gta_2023'

if name =='gta_aggregated':
    g.parse(data="./../data/raw/gta_2021_latest_clean.nt", format = "ntstar") 
elif name == 'gta_2023':
    g.parse(data="./../data/raw/gta_2023_latest_clean.nt", format = "ntstar") 
    # g.parse(data="./../data/raw/gta_mini.nt", format = "ntstar") 
# g.parse(data="test/ntriples-star/ntriples-star-syntax-1.nt", format = "ntstar")
# print(g.serialize(format = "ntstar"))

print("the length of the graph is: ", len(g))

quadruples = []

for s, p, o in  g.triples((None,term.URIRef('https://schema.coypu.org/gta#hasAnnouncementDate'), None)): 
    if 'star' in str(s):
        triple = s
        sub = str(triple._subject)
        pred = str(triple._predicate)
        ob =str(triple._object)
        # timestep = str(o)
        date_o = datetime.strptime(str(o), '%Y-%m-%d').date() # extract the timestamp
        if not '/www.wikidata.org/entity' in sub:
            quadruples.append([sub, pred, ob, date_o,0,0,0])

new_quads = quadruples
print(f'found {len(quadruples)} new relations')

if name == 'gta_2023':
    with open('./../data/gta/gta_2023.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in new_quads:
            writer.writerow(row)
elif name == 'gta_aggregated':
    with open('./../data/gta/gta_newquads.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in new_quads:
            writer.writerow(row)
print('done')