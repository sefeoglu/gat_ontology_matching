
from rdflib import *

import ontospy

class OntologyParser(object):
    
    def __init__(self, rdf_file):
        self.onto = ontospy.Ontospy(rdf_file)
        self.classes = self.__get_classes()
        self.properties = self.__get_properties()
    
    def __get_properties(self):
        properties = []
        for prop in self.onto.all_properties:
            prop = str(prop).split("/")[-1].replace("*>","").split("#")[-1]
            properties.append(prop)
        return properties
   
    def __get_classes(self):
        classes = []
        for concept in self.onto.all_classes:
            concept = str(concept).split("/")[-1].replace("*>","").split("#")[-1]
            classes.append(concept)
        return classes

    def __get_all_triples(self):

        triples = self.onto.sparql("SELECT ?subject ?predicate ?object  WHERE { ?subject ?predicate ?object}")
        return triples

    def __get_inferred_properties(self, class_uri):
        props = self.onto.getInferredPropertiesForClass(class_uri)
        return props