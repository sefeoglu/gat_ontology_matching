import  sys
import os

from rdflib import Graph
from rdflib import *

import ontospy

class OntologyParser(object):
    
    def __init__(self, rdf_file):
        self.onto = ontospy.Ontospy(rdf_file)
        self.classes = self.get_classes()
        self.properties = self.get_properties()
    
    def get_properties(self):
        properties = []
        for prop in self.onto.all_properties:
            prop = str(prop).split("/")[-1].replace("*>","").split("#")[-1]
            properties.append(prop)
        return properties
   
    def get_classes(self):
        classes = []
        for concept in self.onto.all_classes:
            concept = str(concept).split("/")[-1].replace("*>","").split("#")[-1]
            classes.append(concept)
        return classes

    def get_all_triples(self):

        triples = self.onto.sparql("SELECT ?subject ?predicate ?object  WHERE { ?subject ?predicate ?object}")
        return triples

    def get_inferred_properties(self, class_uri):
        props = self.onto.getInferredPropertiesForClass(class_uri)
        return props