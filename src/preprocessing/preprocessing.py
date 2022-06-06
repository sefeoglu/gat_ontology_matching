
import sys, configparser
import os, itertools, re, logging, requests, urllib
import pandas as pd
import networkx as nx
from xml.dom import minidom
import itertools
import numpy as np
import logging
from scipy import spatial

from sentence_transformers import SentenceTransformer

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from OntologyParser import OntologyParser


try:
    # ubuntu works with USE(universal sentence encoder), otherwise it will run with bert sentence encoder
    import tensorflow_hub as hub
except:
    pass

flatten = lambda l: [item for sublist in l for item in sublist]

def cos_sim(a,b):
    return 1 - spatial.distance.cosine(a, b)

class GraphParser(object):

    def __init__(self, ontologies_in_alignment, alignments=None):
        ### Folder Definition ###

        self.prefix_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/graph_representation/"
        config = configparser.ConfigParser()
        config.read(self.prefix_path + 'config.ini')
        if alignments == None:
           self.alignments = [] 
        else:
            self.alignments = alignments
            
        self.ontologies_in_alignment = ontologies_in_alignment
        self.train_folder = self.prefix_path + str(config['Paths']['dataset_folder']) + str(config["General"]['dataset'])+'/ontologies/'
        self.clean_set = self.prefix_path + str(config["Paths"]["spreadsheet_folder"]) + str(config["Paths"]["clean_dataset"])
        self.preprocessed_dataset = self.prefix_path + str(config["Paths"]["spreadsheet_folder"]) + str(config["Paths"]["preprocessed_dataset"])
        self.alignment_folder = self.prefix_path + str(config['Paths']['dataset_folder']) + str(config["General"]['dataset']) + str(config["Paths"]["alignment_folder"])
        self.output_folder_csv = self.prefix_path + str(config["Paths"]["spreadsheet_folder"]) + str(config['Paths']["dataset_folder"])
        self.all_data_folder = self.prefix_path + str(config["Paths"]["spreadsheet_folder"])
        try:
            self.USE_link = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
            self.USE = hub.load(self.USE_link)
        except:
            pass
        self.model_transformer = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.stopwords = ["has"]

    def extractUSEEmbeddings(self, words):
        try:
            word_embeddings = self.USE(words).numpy()
        except:
            word_embeddings = self.model_transformer.encode(words)

        return word_embeddings

    def create_spreadsheet_from_triples(self):
        """Creates spreadsheet from triples
        """        
        file_list = os.listdir(self.train_folder)
        subject, predicate, object_ls, ns_l, ontology_list= [], [], [], [], []
        for file_name in file_list:
            ontology_list.append(self.train_folder+file_name)
            rdf_file = self.train_folder+file_name
            o_parser =  OntologyParser(rdf_file)
            triples = o_parser.get_all_triples()
            ns = file_name.split('.')[0]
            for s, p, o in triples:
                subject.append(s)
                predicate.append(p)
                object_ls.append(o)
                ns_l.append(ns)

        data = {'subject':subject, 'predicate':predicate, 'object': object_ls,'ns':ns_l}
        data_frame = pd.DataFrame(data)

        return data_frame
            
            
    def inferred_properties(self, ontology_parser):

        inferred_properties = []
        classes = ontology_parser.get_classes()
        for ent in classes:
            inferred_properties.append(ontology_parser.get_inferred_properties(ent.split('#')[1]))

        return inferred_properties

    def preprocess_triples(self, data_frame):   

        data_frame["predicate"] = data_frame["predicate"].apply(lambda x: x.split('#')[-1].split("/")[-1].split("pred:")[-1].replace("*>","").split("#")[-1].split("pred:")[-1])

        data_frame["subject"] = data_frame["subject"].apply(lambda x: x.split('#')[-1].split("/")[-1].split("pred:")[-1].replace("*>","").split("#")[-1].split("pred:")[-1])
      
        data_frame["object"] = data_frame["object"].apply(lambda x: x.split('#')[-1].split("/")[-1].split("pred:")[-1].replace("*>","").split("#")[-1].split("pred:")[-1])
      
        return data_frame

    def construct_neighbour_dicts(self,ontologies_in_alignment):

        neighbours_dicts_ent, neighbours_dicts_prop = {}, {}
        flatten = lambda l: [item for sublist in l for item in sublist]
        data_frame = self.data_frame 

        for ont in list(set(flatten(ontologies_in_alignment))):
            
            onto_frame = data_frame[data_frame.ns == ont.split('/')[-1].split('.')[0]]
            self.kg_df = pd.DataFrame({'source':onto_frame.subject, 'target':onto_frame.object, 'edge':onto_frame.predicate})
            
            neighbours_dicts_ent.update(self.entity_neighbour_dict(ont))

            neighbours_dicts_prop.update(self.property_neighbour_dict(ont))

        max_types = np.max([len([nbr_type for nbr_type in elem if flatten(nbr_type)]) for elem in neighbours_dicts_ent.values()])

        return neighbours_dicts_ent, neighbours_dicts_prop, max_types

    def entity_neighbour_dict(self, ontology):
        self.parser = OntologyParser(ontology)
        entities = self.parser.classes
        neighbours_dicts_ent = {}
        ont = str(ontology).split('/')[-1].split('.')[0]

        for ent in entities:
            subcClass_neighbour, equivalent_neighbour, domain_neighbour, range_neighbour, parents = self.one_hop_class(ent, ont)
            ent = ont+"#"+str(ent).split("/")[-1].replace("*>","").split("#")[-1]
            neighbours_dicts_ent[ent] = [[parents], subcClass_neighbour, domain_neighbour, range_neighbour, equivalent_neighbour]

        return neighbours_dicts_ent

    def property_neighbour_dict(self, ontology):
        
        parser = OntologyParser(ontology)
        properties = parser.properties
        neighbours_dicts_prop = {}
        ont = str(ontology).split('/')[-1].split('.')[0]

        for prop in properties:
            domain, _range, subprop, inverse_prop = self.one_hop_property(prop, ont)
            prop_node = ont+"#"+str(prop).split("/")[-1].replace("*>","").split("#")[-1]
            prop = ont+"#"+str(prop).split("/")[-1].replace("*>","").split("#")[-1]
            neighbours_dicts_prop[prop]=[[prop_node], domain, _range, subprop, inverse_prop]

        return neighbours_dicts_prop

    def one_hop_property(self, prop, ont, attention_type='two'):
        # get onProperty
        prop_range, prop_domain = [], []
        somevalues_df = self.kg_df[self.kg_df['edge']=='someValuesFrom'] 
        unionOf_df = self.kg_df[self.kg_df['edge']=='unionOf']
        oneof_df = self.kg_df[self.kg_df['edge']=='oneOf']
        allValuesFrom_df = self.kg_df[self.kg_df['edge']=='allValuesFrom']
        first_df = self.kg_df[self.kg_df['edge']=='first']
        rest_df = self.kg_df[self.kg_df['edge']=='rest']
        domain_df = self.kg_df[self.kg_df['edge']== 'domain']
        range_df = self.kg_df[self.kg_df['edge']== 'range']
       
        G = self.create_graph(domain_df)
        prop = prop.split('#')[-1]
        ### Domain of Property
        try:
            props =  nx.neighbors(G, prop)
            for p in props:
                if str(p).startswith('N'):
                    prop_domain = self.unionOf(p, unionOf_df, first_df, rest_df, prop_domain, ont)
                    prop_domain = self.Restriction(p, somevalues_df, first_df, rest_df, unionOf_df, prop_domain, ont)
                    prop_domain = self.Restriction(p, oneof_df, first_df, rest_df, unionOf_df, prop_domain, ont)
                    prop_domain = self.Restriction(p, allValuesFrom_df, first_df, rest_df, unionOf_df, prop_domain, ont)
                else:
                    prop_domain.append([ont+"#"+p])
        except:
            pass
        ### Range of Property###
        G = self.create_graph(range_df)
        try:
            props =  nx.neighbors(G, prop)
            for p in props:
                if str(p).startswith('N'):
                    prop_range = self.unionOf(p, unionOf_df, first_df, rest_df, prop_range, ont)
                    prop_range = self.Restriction(p, somevalues_df, first_df, rest_df, unionOf_df, prop_range, ont)
                    prop_range = self.Restriction(p, oneof_df, first_df, rest_df, unionOf_df, prop_range, ont)
                    prop_range = self.Restriction(p, allValuesFrom_df, first_df, rest_df, unionOf_df, prop_range, ont)
                else:
                    prop_range.append([ont+"#"+p])
        except:
            pass

        ## For subprop
        subprop, inverse_prop = [], []
        if attention_type == 'two':
            subprop, inverse_prop = self.two_hop_property(prop, subprop, inverse_prop, ont)
        
        if len(prop_domain) != 0:
            prop_domain =  prop_domain[0]
        if len(prop_range) != 0:
            prop_range =  prop_range[0]

        if len(subprop) != 0:
            subprop =  subprop[0]
        if len(inverse_prop) != 0:
            inverse_prop =  inverse_prop[0]

        return prop_domain, prop_range, subprop, inverse_prop

    def create_graph(self, data_frame):
        G=nx.from_pandas_edgelist(data_frame, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
        G = nx.to_undirected(G)
        return G

    def Restriction(self,node, main_df,first_df, rest_df, union_df, class_neighbour, ont):
        try:
            G = self.create_graph(main_df)
            allnodes = nx.neighbors(G, node)
            for anode in allnodes:
                if str(anode).startswith('N'):
                    class_neighbour = self.unionOf(anode,union_df,first_df, rest_df, class_neighbour, ont)
                    try:
                        G = self.create_graph(first_df)
                        first_nodes = nx.neighbors(G, anode)
                        for fn in first_nodes:
                            if str(fn).startswith('N'):
                                continue
                            else:
                                class_neighbour.append([ont+"#"+fn])
                        G = self.create_graph(rest_df)
                        rest_nodes = nx.neighbors(G, anode)
                        for rn in rest_nodes:
                            if str(rn).startswith('N'):
                                try:
                                    G = self.create_graph(first_df)
                                    fnodes2 = nx.neighbors(G, rn)
                                    for fn2 in fnodes2:
                                        class_neighbour.append([ont+"#"+fn2])
                                except:
                                    pass
                            else:
                                class_neighbour.append([ont+"#"+fn])
                    except:
                        pass

                else:
                    class_neighbour.append([ont+"#"+anode])
        except:
            pass
        return class_neighbour



    def unionOf(self, node, unionOf_df, first_df,rest_df, class_neighbour, ont):
        try:
            G = self.create_graph(unionOf_df)
            unodes = nx.neighbors(G, node)
            for unode in unodes:
                if str(unode).startswith('N'):
                    try:
                        G = self.create_graph(first_df)
                        first_nodes = nx.neighbors(G, unode)
                        for fn in first_nodes:
                            if str(fn).startswith('N'):
                                continue
                            else:
                                class_neighbour.append([ont+"#"+fn])
                        G = self.create_graph(rest_df)
                        rest_nodes = nx.neighbors(G, unode)
                        for rn in rest_nodes:
                            if str(rn).startswith('N'):
                                try:
                                    G = self.create_graph(first_df)
                                    fnodes2 = nx.neighbors(G, rn)
                                    for fn2 in fnodes2:
                                        class_neighbour.append([ont+"#"+fn2])
                                    # Rest of Rest
                                    G = self.create_graph(rest_df)
                                    rnodes2 = nx.neighbors(G, rn)
                                    for rn2 in rnodes2:
                                        if str(rn2).startswith('N'):
                                            try:
                                                G = self.create_graph(first_df)
                                                fnodes3 = nx.neighbors(G, rn2)
                                                for fn3 in fnodes3:
                                                    class_neighbour.append([ont+"#"+fn3])
                                            except:
                                                pass
                                        else:
                                            class_neighbour.append([ont+"#"+rn2])
                                except:
                                    pass
                            else:
                                class_neighbour.append([ont+"#"+fn])
                    except:
                        pass
                else:
                    class_neighbour.append([ont+"#"+unode])
        except:
            pass

        return class_neighbour


    def onProperty(self, rsnode, onProperty_df, class_neighbour, ont):

        try:
            G = self.create_graph(onProperty_df)
            nodes = nx.neighbors(G, rsnode)
            for node in nodes:
                class_neighbour.append([ont+"#"+node])
        except:
            pass

        return class_neighbour


    def range_of_prop(self, prop_node, range_df, class_neighbour, ont):
        try:
            G = self.create_graph(range_df)
            nodes = nx.neighbors(G, prop_node)
            for node in nodes:
                class_neighbour.append([ont+"#"+node])
        except:
            pass

        return class_neighbour
    

    def one_hop_class(self, entity, ont):
        ## get subClassOf and equivalentClass
        class_df = self.kg_df[self.kg_df['edge']=='subClassOf'] 
        equivalent_df = self.kg_df[self.kg_df['edge']=='equivalentClass']
        somevalues_df = self.kg_df[self.kg_df['edge']=='someValuesFrom'] 
        unionOf_df = self.kg_df[self.kg_df['edge']=='unionOf']
        oneof_df = self.kg_df[self.kg_df['edge']=='oneOf']
        allValuesFrom_df = self.kg_df[self.kg_df['edge']=='allValuesFrom']
        first_df = self.kg_df[self.kg_df['edge']=='first']
        rest_df = self.kg_df[self.kg_df['edge']=='rest']
        domain_df = self.kg_df[self.kg_df['edge']== 'domain']
        onProperty_df = self.kg_df[self.kg_df['edge']=='onProperty']
        range_df = self.kg_df[self.kg_df['edge']=='range']
        
        G = self.create_graph(class_df)
        subcClass_neighbour, domain_neighbour, equivalent_neighbour, range_neighbour = [], [], [], []
        ### subClassOf for children ###
        try:
            nodes = nx.neighbors(G, entity)
            for node in nodes:
                if str(node).startswith('N'):
                    subcClass_neighbour = self.unionOf(node, unionOf_df, first_df, rest_df, subcClass_neighbour, ont)
                    subcClass_neighbour = self.Restriction(node, somevalues_df, first_df, rest_df, unionOf_df, subcClass_neighbour, ont)
                    subcClass_neighbour = self.Restriction(node, oneof_df, first_df, rest_df, unionOf_df, subcClass_neighbour, ont)
                    subcClass_neighbour = self.Restriction(node, allValuesFrom_df, first_df, rest_df, unionOf_df, subcClass_neighbour, ont)
                    subcClass_neighbour = self.onProperty(node, onProperty_df, subcClass_neighbour, ont)
                else:
                    subcClass_neighbour.append([ont+"#"+node])
        except:
            pass

        ## path to root
        G=nx.from_pandas_edgelist(class_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
        all_parents = self.parser.onto.toplayer_classes
        parents = [] 
        for parent in all_parents:
            parent = str(parent).split("*")[1].replace("*>","").split("#")[-1].split("/")[-1]
            try:
                nodes = nx.astar_path(G, entity, parent)
                for node in nodes:
                    if str(node).startswith('N'):
                        parents = self.unionOf(node, unionOf_df, first_df, rest_df, parents, ont)
                        parents = self.Restriction(node, somevalues_df, first_df, rest_df, unionOf_df, parents, ont)
                        parents = self.Restriction(node, oneof_df, first_df, rest_df, unionOf_df, parents, ont)
                        parents = self.Restriction(node, allValuesFrom_df, first_df, rest_df, unionOf_df, parents, ont)
                        parents = self.onProperty(node, onProperty_df, parents, ont)
                    else:
                        parents.append([ont+"#"+node])
            except:
                pass
        
        if len(parents)!= 0:
            parent_formatted = []
            for parent in parents:
                parent_formatted.append(parent[0])
            parents = parent_formatted
        ## remove the parent class from children class
        for child_class in subcClass_neighbour:
            child_str = child_class[0]
            for parent in parents:
                if child_str == parent and len(subcClass_neighbour) != 0:
                    subcClass_neighbour.remove(child_class)
                    
        ####equivalentClass
        G = self.create_graph(equivalent_df)
        try:
            nodes = nx.neighbors(G, entity)
            for node in nodes:
                if str(node).startswith('N'):
                    equivalent_neighbour = self.unionOf(node, unionOf_df, first_df, rest_df, equivalent_neighbour, ont)
                    equivalent_neighbour = self.Restriction(node, somevalues_df, first_df, rest_df, unionOf_df, equivalent_neighbour, ont)
                    equivalent_neighbour = self.Restriction(node, oneof_df, first_df, rest_df, unionOf_df, equivalent_neighbour, ont)
                    equivalent_neighbour = self.Restriction(node, allValuesFrom_df, first_df, rest_df, unionOf_df, equivalent_neighbour, ont)
                    ## onProperty
                    equivalent_neighbour = self.onProperty(node, onProperty_df, equivalent_neighbour, ont)
                else:
                    equivalent_neighbour.append([ont+"#"+node])
        except:
            pass

        ## Domain and Range for props
        G = self.create_graph(range_df)
        try:
            nodes = nx.neighbors(G, entity)
            for node in nodes:
                range_neighbour.append([ont+"#"+node])
                try:
                    G = self.create_graph(domain_df)
                    domain_nodes = nx.neighbors(G, node)
                    for dn in domain_nodes:
                        if str(dn).startswith('N'):
                            range_neighbour = self.unionOf(dn, unionOf_df, first_df, rest_df, range_neighbour, ont) 
                            range_neighbour = self.Restriction(dn, somevalues_df, first_df, rest_df, unionOf_df, range_neighbour, ont)
                            range_neighbour = self.Restriction(dn, oneof_df, first_df, rest_df, unionOf_df, range_neighbour, ont)
                            range_neighbour = self.Restriction(dn, allValuesFrom_df, first_df, rest_df, unionOf_df, range_neighbour, ont)
                        else:
                            range_neighbour.append([ont+"#"+dn])
                except:
                    pass
                #find range of node
        except:
            pass
        ## domain probs
         
        G = self.create_graph(domain_df)
        try:
            nodes = nx.neighbors(G, entity)
            for node in nodes:
                domain_neighbour.append([ont+"#"+node])
                try:
                    G = self.create_graph(range_df)
                    range_nodes = nx.neighbors(G, node)
                    for rn in range_nodes:
                        if str(rn).startswith("N"):
                            domain_neighbour = self.unionOf(rn, unionOf_df, first_df, rest_df, domain_neighbour, ont)
                            domain_neighbour = self.Restriction(rn, somevalues_df, first_df, rest_df, unionOf_df, domain_neighbour, ont)
                            domain_neighbour = self.Restriction(rn, oneof_df, first_df, rest_df, unionOf_df, domain_neighbour, ont)
                            domain_neighbour = self.Restriction(rn, allValuesFrom_df, first_df, rest_df, unionOf_df, domain_neighbour, ont)
                        else:
                            domain_neighbour.append([ont+"#"+rn])
                except:
                    pass
                #find range of node
        except:
            pass
        
        return subcClass_neighbour, equivalent_neighbour, domain_neighbour, range_neighbour, parents


    def two_hop_property(self, prop_node, subprop, inverse_prop, ont):

        subProperty_df = self.kg_df[self.kg_df['edge']== 'subPropertyOf']
        inverse_df = self.kg_df[self.kg_df['edge']== 'inverseOf']
        ##subPropertyOf
        try:
            G = self.create_graph(subProperty_df)
            nodes = nx.neighbors(G, prop_node)
            for node in nodes:
                subprop.append([ont+"#"+node])
        except:
            pass
        ### inverseOf
        try:
            G = self.create_graph(inverse_df)
            nodes = nx.neighbors(G, prop_node)
            for node in nodes:
                inverse_prop.append([ont+"#"+node])
        except:
            pass
        return subprop, inverse_prop


    def cleaning_ontology_elements(self, data_frame):
        """ Clean data from ontology elements without semantic relations regarding concepts
        """        
        full_data = self.object_cleanup(data_frame)
        full_data = self.predicate_cleanup(full_data)
        full_data.drop_duplicates(keep='first',inplace=True) 

        return full_data


    def object_cleanup(self, full_data):
        
        full_data = full_data[full_data.object.str.startswith('nil') == False]
        full_data = full_data[full_data.object.str.startswith('All') == False]
        full_data = full_data[full_data.object.str.startswith('Ontology') == False]
        full_data = full_data[full_data.object.str.startswith('Thing') == False]
        # Restriction
        full_data = full_data[full_data.object.str.startswith('Rest') == False]
        
        return full_data


    def predicate_cleanup(self,full_data ):
        predicate_eliminator = ['cardinality', 'comment','complementOf','maxCardinality', 'minCardinality',
                       'qualifiedCardinality','versionInfo', 'hasValue', 'disjointWith']
        clean_full_data = full_data
        for eliminator in predicate_eliminator:
            clean_full_data = clean_full_data[clean_full_data.predicate != eliminator]
        # There is a problem regarding hasValue.
        clean_full_data = clean_full_data[clean_full_data.predicate != 'hasValue']
        return clean_full_data


    def cleaning_process(self):
        data_frame = self.create_spreadsheet_from_triples()
        data_frame = self.preprocess_triples(data_frame)
        self.data_frame = self.cleaning_ontology_elements(data_frame)

        return self.data_frame


    def generate_mappings(self, ontologies_in_alignment, gt_mappings):
        
        ent_mappings, prop_mappings = [], []

        for l in ontologies_in_alignment:

            ont1 = l[0]
            ont2 = l[1]
            parser1 = OntologyParser(ont1)
            parser2 = OntologyParser(ont2)
            ent1 = parser1.classes
            ent2 = parser2.classes
            prop1 = parser1.properties
            prop2 = parser2.properties
            ent_mapping = list(itertools.product(ent1, ent2))

            prop_mapping = list(itertools.product(prop1, prop2))

            pre1 = l[0].split("/")[-1].rsplit(".",1)[0].replace("-", "_")
            pre2 = l[1].split("/")[-1].rsplit(".",1)[0].replace("-", "_")

            ent_mappings.extend([(str(pre1) + "#" + str(el[0]), str(pre2) + "#" + str(el[1])) for el in ent_mapping])
            prop_mappings.extend([(str(pre1) + "#" + str(el[0]), str(pre2) + "#" + str(el[1])) for el in prop_mapping])

        if gt_mappings:
            data_ent = {mapping: False for mapping in ent_mappings}
            data_prop = {mapping: False for mapping in prop_mappings}
            s_ent = set(ent_mappings)
            s_prop = set(prop_mappings)

            for mapping in set(gt_mappings):
                if mapping in s_ent:
                    data_ent[mapping] = True
                elif mapping in s_prop:
                    data_prop[mapping] = True
                else:
                    mapping = tuple([el.replace(",-", "_") for el in mapping])
                    if mapping in s_ent:
                        data_ent[mapping] = True
                    elif mapping in s_prop:
                        data_prop[mapping] = True
                    else:
                        logging.info ("Warning: {} given in alignments could not be found in source/target ontology.".format(mapping))
                        continue
            return (data_ent, data_prop)

        return (ent_mappings, prop_mappings)


    def run_abbreviation_resolution(self, inp, filtered_dict):
        # Resolving abbreviations to full forms
        logging.info ("Resolving abbreviations...")
        inp_resolved = []
        for concept in inp:
            for key in filtered_dict:
                concept = concept.replace(key, filtered_dict[key])
            final_list = []
            # Lowering case except in abbreviations
            for word in concept.split(" "):
                if not re.search("[A-Z][A-Z]+", word):
                    final_list.append(word.lower())
                else:
                    final_list.append(word)
            concept = " ".join(final_list)
            inp_resolved.append(concept)

        return inp_resolved



    def run_spellcheck(self, inp_resolved):
        # Spelling checker and corrector
        logging.info ("Running spellcheck...")

        url = "https://grammarbot.p.rapidapi.com/check"

        headers = {
            'x-rapidapi-host': "grammarbot.p.rapidapi.com",
            'x-rapidapi-key': "9965b01207msh06291e57d6f2c55p1a6a16jsn0fb016da4a62",
            'content-type': "application/x-www-form-urlencoded"
            }

        inp_spellchecked = []

        for concept in inp_resolved:
            payload = "language=en-US&text=" + urllib.parse.quote_plus(concept)
            response = requests.request("POST", url, data=payload, headers=headers).json()
            concept_corrected = str(concept)
            
            for elem in response["matches"]:
                start, end = elem["offset"], elem["offset"] + elem["length"]
                concept_corrected = concept_corrected[:start] + elem["replacements"][0]["value"] + concept_corrected[end:]
            
            if concept.lower() != concept_corrected.lower():
                logging.info ("{} corrected to {}".format(concept, concept_corrected))
                inp_spellchecked.append(concept_corrected)
            else:
                inp_spellchecked.append(concept)

        return inp_spellchecked



    def construct_abbreviation_resolution_dict(self, all_mappings):
        # Constructs an abbrevation resolution dict
        logging.info ("Constructing abbrevation resolution dict....")
    
        final_dict = {}
        for mapping in all_mappings:
            mapping = tuple([ str(el).split("#")[1] for el in mapping])
            is_abb = re.search("[A-Z][A-Z]+", mapping[0])
            if is_abb:
                abbreviation = "".join([el[0].upper() for el in mapping[1].split("_")])
                if is_abb.group() in abbreviation:
                    
                    start = abbreviation.find(is_abb.group())
                    end = start + len(is_abb.group())
                    fullform = "_".join(mapping[1].split("_")[start:end])
                    
                    rest_first = " ".join([el for el in mapping[0].replace(is_abb.group(), "").split("_") if el]).lower()
                    rest_second = " ".join(mapping[1].split("_")[:start] + mapping[1].split("_")[end:])
                    if is_abb.group() not in final_dict:
                        final_dict[is_abb.group()] = [(fullform, rest_first, rest_second)]
                    else:
                        final_dict[is_abb.group()].append((fullform, rest_first, rest_second))

            is_abb = re.search("[A-Z][A-Z]+", mapping[1])
            if is_abb:
                abbreviation = "".join([el[0].upper() for el in mapping[0].split("_")])
                
                if is_abb.group() in abbreviation:
                    start = abbreviation.find(is_abb.group())
                    end = start + len(is_abb.group())
                    fullform = "_".join(mapping[0].split("_")[start:end])

                    rest_first = " ".join([el for el in mapping[1].replace(is_abb.group(), "").split("_") if el]).lower()
                    rest_second = " ".join(mapping[0].split("_")[:start] + mapping[0].split("_")[end:])
                    if is_abb.group() not in final_dict:
                        final_dict[is_abb.group()] = [(fullform, rest_first, rest_second)]
                    else:
                        final_dict[is_abb.group()].append((fullform, rest_first, rest_second))

        keys = [el for el in list(set(flatten([flatten([tup[1:] for tup in final_dict[key]]) for key in final_dict]))) if el]
        abb_embeds = dict(zip(keys, self.extractUSEEmbeddings(keys)))

        scored_dict = {}
        for abbr in final_dict:
            sim_list = [(tup[0], tup[1], tup[2], cos_sim(abb_embeds[tup[1]], abb_embeds[tup[2]])) if tup[1] and tup[2]
                        else (tup[0], tup[1], tup[2], 0) for tup in final_dict[abbr]]
            scored_dict[abbr] = sorted(list(set(sim_list)), key=lambda x:x[-1], reverse=True)

        resolved_dict = {key: scored_dict[key][0] for key in scored_dict}
        filtered_dict = {key: " ".join(resolved_dict[key][0].split("_")) for key in resolved_dict if resolved_dict[key][-1] > 0.9}
        logging.info ("Results after abbreviation resolution: ", filtered_dict)
        return filtered_dict



    def camel_case_split(self, identifier):
        # Splits camel case strings
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]



    def parse(self, word):
        return " ".join(flatten([el.split("_") for el in self.camel_case_split(word)]))



    def extract_keys(self, ontologies_in_alignment):

        extracted_elems = []
        data_frame = self.data_frame 
        for ont_name in list(set(flatten(ontologies_in_alignment))):
            ont_name_filt = ont_name.split("/")[-1].rsplit(".",1)[0].replace("-", "_")
            onto_frame = data_frame[data_frame.ns == ont_name_filt]
            self.predicate = list(onto_frame["predicate"])
            self.subject = list(onto_frame["subject"])
            self.object =list(onto_frame["object"])
            extracted_elems.extend([ont_name_filt + "#" + elem for elem in self.subject + self.predicate + self.object])
        extracted_elems = list(set(extracted_elems))
        inp = []
        for word in extracted_elems:
            ont_name = word.split("#")[0]
            elem = word.split("#")[1]
            inp.append(self.parse(elem))

        logging.info ("Total number of extracted unique classes and properties from entire RA set: ", len(extracted_elems))

        extracted_elems = ["<UNK>"] + extracted_elems
        return inp, extracted_elems



    def extract_embeddings(self, inp, extracted_elems):

        # Creates embeddings to index dict, word to index dict etc
        embeds = np.array(self.extractUSEEmbeddings(inp))
        embeds = np.array([np.zeros(embeds.shape[1],)] + list(embeds))
        embeddings = dict(zip(extracted_elems, embeds))
        emb_vals = list(embeddings.values())
        emb_indexer = {key: i for i, key in enumerate(list(embeddings.keys()))}
        emb_indexer_inv = {i: key for i, key in enumerate(list(embeddings.keys()))}

        return emb_vals, emb_indexer, emb_indexer_inv


    def remove_stopwords(self, inp):
        # Remove high frequency stopwords
        inp_filtered = []
        for elem in inp:
            words = " ".join([word for word in elem.split() if word not in self.stopwords])
            words = words.replace("-", " ")
            inp_filtered.append(words)
        return inp_filtered


    def process(self, spellcheck=False):
        self.data_frame = self.cleaning_process()
        gt_mappings = [tuple([elem.split("/")[-1] for elem in el]) for el in self.alignments]
        gt_mappings = [tuple([el.split("#")[0].rsplit(".", 1)[0] +  "#" +  el.split("#")[1] for el in tup]) for tup in gt_mappings]

        ent_mappings, prop_mappings = self.generate_mappings(self.ontologies_in_alignment, gt_mappings) #OK
        
        inp, extracted_elems = self.extract_keys(self.ontologies_in_alignment)
        filtered_dict = self.construct_abbreviation_resolution_dict(list(ent_mappings) + list(prop_mappings))

        inp_resolved = self.run_abbreviation_resolution(inp, filtered_dict)
        if spellcheck:
            try:
                inp_resolved = self.run_spellcheck(inp_resolved)
            except:
                pass
        inp = self.remove_stopwords(inp_resolved)
        emb_vals, emb_indexer, emb_indexer_inv = self.extract_embeddings(inp, extracted_elems)
        neighbours_dicts_ent, neighbours_dicts_prop, max_types = self.construct_neighbour_dicts(self.ontologies_in_alignment)
   
        return  ent_mappings, prop_mappings, emb_indexer, emb_indexer_inv, emb_vals, neighbours_dicts_ent, neighbours_dicts_prop, max_types
