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
# source of rdflib: https://github.com/XuguangSong98/rdflib

g = Graph()

name = 'acled_collapsed'

if name =='acled_collapsed':
    # g.parse(data="./../data/raw/gta_2021_latest_clean.nt", format = "ntstar") 
    g.parse(data="./../data/raw/acled_2023_collapsed_event_types.nt", format = "ntstar") 
# g.parse(data="test/ntriples-star/ntriples-star-syntax-1.nt", format = "ntstar")
# print(g.serialize(format = "ntstar"))

print("the length of the graph is: ", len(g))

quadruples = []

for s, p, o in  g.triples((None,term.URIRef('https://schema.coypu.org/global#hasTimestamp'), None)): 
    if 'star' in str(s):
        triple = s
        sub = str(triple._subject)
        pred = str(triple._predicate)
        ob =str(triple._object)
        # timestep = str(o)
        date_o = datetime.strptime(str(o), '%Y-%m-%d').date() # extract the timestamp
        quadruples.append([sub, pred, ob, date_o,0,0,1])

new_quads = quadruples
print(f'found {len(quadruples)} new relations')

with open('./../data/acled/acled_2023_event_types.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in new_quads:
        writer.writerow(row)

print('done')