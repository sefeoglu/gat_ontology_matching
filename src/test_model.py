import configparser, sys, logging, random, os, pickle
import numpy as np
from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ontology import *
PACKAGE_PARENT = '..'

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from graph_representation.preprocessing.preprocessing import *
from graph_representation.preprocessing.preprocessing import GraphParser

from model.HAN import SiamHAN



 sys.argv[1], sys.argv[2]

#ont_name1, ont_name2 = doc1, doc2
if ont_name1.endswith("/"):
    ont_name1 = ont_name1[:-1]
if ont_name2.endswith("/"):
    ont_name2 = ont_name2[:-1]

PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read(PREFIX_PATH + 'config.ini')

logging.info("Prefix path: ", PREFIX_PATH)

# Initialize variables from config
quick_mode = str(config["General"]["quick_mode"])

model_path = PREFIX_PATH + str(config["Paths"]["load_model_path"])
output_path = PREFIX_PATH + str(config["Paths"]["output_folder"])
cached_embeddings_path = PREFIX_PATH + str(config["Paths"]["embedding_cache_path"])
spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

max_paths = int(config["Parameters"]["max_paths"])
max_pathlen = int(config["Parameters"]["max_pathlen"])


batch_size = int(config["Hyperparameters"]["batch_size"])

test_ontologies = [tuple([ont_name1, ont_name2])]

# Preprocessing and parsing input data for testing

preprocessing = GraphParser(test_ontologies)
test_data_ent, test_data_prop, emb_indexer_new, emb_indexer_inv_new, emb_vals_new, neighbours_dicts_ent, neighbours_dicts_prop, max_types = preprocessing.process(spellcheck)

if os.path.isfile(cached_embeddings_path):
    logging.info("Found cached embeddings...")
    emb_indexer_cached, emb_indexer_inv_cached, emb_vals_cached = pickle.load(open(cached_embeddings_path, "rb"))
else:
    emb_indexer_cached, emb_indexer_inv_cached, emb_vals_cached = {}, {}, []

emb_vals, emb_indexer, emb_indexer_inv = list(emb_vals_cached), dict(emb_indexer_cached), dict(emb_indexer_inv_cached)

s = set(emb_indexer.keys())
idx = len(emb_indexer_inv)
for term in emb_indexer_new:
    if term not in s:
        emb_indexer[term] = idx
        emb_indexer_inv[idx] = term
        emb_vals.append(emb_vals_new[emb_indexer_new[term]])
        idx += 1

def count_non_unk(elem):
    return len([l for l in elem if l!="<UNK>"])

def generate_data_neighbourless(elem_tuple):
    return [emb_indexer[elem] for elem in elem_tuple]

def embedify(seq):
    for item in seq:
        if isinstance(item, list):
            yield list(embedify(item))
        else:
            yield emb_indexer[item]

def generate_data(elem_tuple, neighbours_dicts):
    return list(embedify([neighbours_dicts[elem] for elem in elem_tuple]))

def to_feature(inputs):
    inputs_lenpadded = [[[[path[:max_pathlen] + [0 for i in range(max_pathlen -len(path[:max_pathlen]))]
                                    for path in nbr_type[:max_paths]]
                                for nbr_type in ent[:max_types]]
                            for ent in elem]
                        for elem in inputs]
    inputs_pathpadded = [[[nbr_type + [[0 for j in range(max_pathlen)]
                             for i in range(max_paths - len(nbr_type))]
                            for nbr_type in ent] for ent in elem]
                        for elem in inputs_lenpadded]
    return inputs_pathpadded

def pad_prop(inputs):
    inputs_padded = [[[elem + [0 for i in range(max_prop_len - len(elem))]
                         for elem in prop]
                    for prop in elem_pair]
                for elem_pair in inputs]
    return inputs_padded

def generate_input(elems, neighbours_dicts):
    inputs, nodes = [], []
    
    global DIRECT_INPUTS
    for elem in list(elems):
        try:
            inputs.append(generate_data(elem, neighbours_dicts))
            nodes.append(generate_data_neighbourless(elem))
        except Exception as e:
            DIRECT_INPUTS.append(generate_data_neighbourless(elem))
    return inputs, nodes

def write_results():
    ont_name_parsed1 = Ontology(ont_name1).extract_ns()
    ont_name_parsed2 = Ontology(ont_name2).extract_ns()
    ont_name1_pre = ont_name1 if (ont_name1.startswith("http://") or ont_name1.startswith("https://")) else "file://" + ont_name1
    ont_name2_pre = ont_name2 if (ont_name2.startswith("http://") or ont_name2.startswith("https://")) else "file://" + ont_name2
    rdf = \
    """<?xml version='1.0' encoding='utf-8' standalone='no'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'
         xmlns:align='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'>
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>**</type>
  <onto1>
    <Ontology rdf:about="{}">
      <location>{}</location>
    </Ontology>
  </onto1>
  <onto2>
    <Ontology rdf:about="{}">
      <location>{}</location>
    </Ontology>
  </onto2>""".format(ont_name_parsed1.split("#")[0], ont_name1_pre, ont_name_parsed2.split("#")[0], ont_name2_pre)
    for (a,b,score) in final_list:
        mapping = """
  <map>
    <Cell>
      <entity1 rdf:resource='{}'/>
      <entity2 rdf:resource='{}'/>
      <relation>=</relation>
      <measure rdf:datatype='http://www.w3.org/2001/XMLSchema#float'>{}</measure>
    </Cell>
  </map>""".format(ont_name_parsed1 + "#".join(a.split("#")[1:]), ont_name_parsed2 + "#".join(b.split("#")[1:]), score)
        rdf += mapping
    rdf += """
</Alignment>
</rdf:RDF>"""
    return rdf

torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

np.random.shuffle(test_data_ent)
np.random.shuffle(test_data_prop)

torch.set_default_dtype(torch.float64)

logging.info ("Loading trained model....")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_dict = torch.load(model_path, map_location=torch.device(device))

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k!="name_embedding.weight"}
#max_types = len([key for key in pretrained_dict.keys() if key.startswith("w_")]) + 1

model = SiamHAN(emb_vals, max_types, max_paths, max_pathlen).to(device)

model_dict = model.state_dict()

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

threshold = model.threshold.data.cpu().numpy()[0]
print(threshold)

logging.info ("Model loaded successfully!")

print("Optimum Threshold: {}".format(threshold))

model.eval()

logging.info ("Length of test data(ent): {} test data(prop):{}".format(len(test_data_ent), len(test_data_prop)))

all_results = OrderedDict()    
DIRECT_INPUTS = []
with torch.no_grad():
    inputs_all_ent, nodes_all_ent = generate_input(test_data_ent, neighbours_dicts_ent)
    inputs_all_prop, nodes_all_prop = generate_input(test_data_prop, neighbours_dicts_prop)
    all_inp = list(zip(inputs_all_ent, nodes_all_ent))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))

    inputs_all_ent, nodes_all_ent = list(zip(*all_inp_shuffled))

    all_inp = list(zip(inputs_all_prop, nodes_all_prop))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))
    inputs_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled))
    
    max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair]
        for elem_pair in inputs_all_prop])
    logging.info ("Max prop len: ", max_prop_len)
    batch_size = min(batch_size, len(inputs_all_ent))
    num_batches = int(ceil(len(inputs_all_ent)/batch_size))
    batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

    logging.info ("Num batches: {} Batch size (prop): {}".format(num_batches, batch_size_prop))
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        batch_start_prop = batch_idx * batch_size_prop
        batch_end_prop = (batch_idx+1) * batch_size_prop

        inputs_ent = np.array(to_feature(inputs_all_ent[batch_start: batch_end]))
        nodes_ent = np.array(nodes_all_ent[batch_start: batch_end])

        inputs_prop = np.array(pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop]))
        nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])

        inp_ents = torch.LongTensor(inputs_ent).to(device)
        node_ents = torch.LongTensor(nodes_ent).to(device)
        inp_props = torch.LongTensor(inputs_prop).to(device)
        node_props = torch.LongTensor(nodes_prop).to(device)
        outputs = model(node_ents, inp_ents, node_props, inp_props, max_prop_len)
        outputs = [el.item() for el in outputs]

        for idx, pred_elem in enumerate(outputs):
            if idx < len(nodes_ent):
                ent1 = emb_indexer_inv[nodes_ent[idx][0]]
                ent2 = emb_indexer_inv[nodes_ent[idx][1]]
            else:
                ent1 = emb_indexer_inv[nodes_prop[idx-len(nodes_ent)][0]]
                ent2 = emb_indexer_inv[nodes_prop[idx-len(nodes_ent)][1]]
            if (ent1, ent2) in all_results:
                logging.info ("Error: ", ent1, ent2, "already present")
            all_results[(ent1, ent2)] = (round(pred_elem, 3), pred_elem>=threshold)
    
    logging.info ("Len (direct inputs): ", len(DIRECT_INPUTS))
    for idx, direct_input in enumerate(DIRECT_INPUTS):
        ent1 = emb_indexer_inv[direct_input[0]]
        ent2 = emb_indexer_inv[direct_input[1]]
        sim =  cos_sim(emb_vals[direct_input[0]], emb_vals[direct_input[1]])
        all_results[(ent1, ent2)] = (round(sim, 3), pred_elem>=threshold)
    
final_list = [(elem[0], elem[1], str(all_results[elem][0])) for elem in all_results if all_results[elem][1]]

ont_name_parsed1 = Ontology(ont_name1).extract_ns().split("/")[-1].split("#")[0].rsplit(".", 1)[0]
ont_name_parsed2 = Ontology(ont_name2).extract_ns().split("/")[-1].split("#")[0].rsplit(".", 1)[0]

f = "HAN-"+ont_name_parsed1.lower() + "-" + ont_name_parsed2.lower() + ".rdf"

open(output_path + f, "w+").write(write_results())

logging.info ("The final alignment file can be found below: ")
print("file://" + output_path + f)
