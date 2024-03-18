# """/*
#  *    Dynamic Representations of Global Crises: A Temporal Knowledge Graph For Conflicts, Trade and Value Networks
#  *
#  *        File: ts_assignment_gta_star_event.py
#  *
#  *     Authors: Deleted for purposes of anonymity 
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  * 
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for 
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software. 
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
#  * 
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  * 
#  * For purposes of anonymity, the identity of the proprietor is not given herewith. 
#  * The identity of the proprietor will be given once the review of the 
#  * conference submission is completed. 
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */"""


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