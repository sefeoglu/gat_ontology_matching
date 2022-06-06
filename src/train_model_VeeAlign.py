import configparser, logging, random, sys, os, pickle
import numpy as np
from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from scipy import spatial
from xml.dom import minidom
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from preprocessing.preprocessing import GraphParser
from VeeAlign.vee_align import VeeAlign

DIRECT_INPUTS, DIRECT_TARGETS = [], []
threshold_results = {}

class Trainer(object):

    def __init__(self):
        PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

        print ("Prefix path: ", PREFIX_PATH)

        # Read `config.ini` and initialize parameter values
        config = configparser.ConfigParser()
        config.read(PREFIX_PATH + 'config.ini')

        # Initialize variables from config

        self.quick_mode = str(config["General"]["quick_mode"])

        self.K = int(config["General"]["K"])
        self.ontology_split = str(config["General"]["ontology_split"]) == "True"
        self.max_false_examples = int(config["General"]["max_false_examples"])

        #self.alignment_folder = PREFIX_PATH + "datasets/" + str(config["General"]["dataset"]) + "/alignments/"
        self.train_folder = PREFIX_PATH + "datasets/" + str(config["General"]["dataset"]) + "/ontologies/"
        self.cached_embeddings_path = PREFIX_PATH + str(config["Paths"]["embedding_cache_path"])
        self.model_path = PREFIX_PATH + str(config["Paths"]["save_model_path"])
        self.alignment_folder = PREFIX_PATH + str(config['Paths']['dataset_folder']) + str(config["General"]['dataset']) + str(config["Paths"]["alignment_folder"])
        self.spellcheck = config["Preprocessing"]["has_spellcheck"] == "True"

        self.max_paths = int(config["Parameters"]["max_paths"])
        self.max_pathlen = int(config["Parameters"]["max_pathlen"])
        self.bag_of_neighbours = config["Parameters"]["bag_of_neighbours"] == "True"
        self.weighted_sum = config["Parameters"]["weighted_sum"] == "True"
        ## hyperparamaters
        self.lr = float(config["Hyperparameters"]["lr"])
        self.num_epochs = int(config["Hyperparameters"]["num_epochs"])
        self.weight_decay = float(config["Hyperparameters"]["weight_decay"])
        self.batch_size = int(config["Hyperparameters"]["batch_size"])
        
    def train(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VeeAlign(self.emb_vals, self.max_types, self.max_paths, self.max_pathlen, self.weighted_sum).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        global DIRECT_INPUTS, DIRECT_TARGETS 
        global threshold_results 
        print ("Starting training...")
        epoch_list = []
        loss_list = []
        threshold_list = []
        for epoch in range(self.num_epochs):

            inputs_pos_ent, nodes_pos_ent, targets_pos_ent = self.generate_input(self.train_data_t_ent, 1, self.neighbours_dicts_ent, self.emb_indexer)
            inputs_neg_ent, nodes_neg_ent, targets_neg_ent = self.generate_input(self.train_data_f_ent, 0, self.neighbours_dicts_ent, self.emb_indexer)
            inputs_pos_prop, nodes_pos_prop, targets_pos_prop = self.generate_input(self.train_data_t_prop, 1, self.neighbours_dicts_prop, self.emb_indexer)
            inputs_neg_prop, nodes_neg_prop, targets_neg_prop = self.generate_input(self.train_data_f_prop, 0, self.neighbours_dicts_prop, self.emb_indexer)

            inputs_all_ent = list(inputs_pos_ent) + list(inputs_neg_ent)
            targets_all_ent = list(targets_pos_ent) + list(targets_neg_ent)
            nodes_all_ent = list(nodes_pos_ent) + list(nodes_neg_ent)
            
            all_inp_ent = list(zip(inputs_all_ent, targets_all_ent, nodes_all_ent))
            all_inp_shuffled_ent = random.sample(all_inp_ent, len(all_inp_ent))
            inputs_all_ent, targets_all_ent, nodes_all_ent = list(zip(*all_inp_shuffled_ent))

            inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
            targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
            nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)

            if len(inputs_all_prop) == 0:
                max_prop_len = 0
            else:
                max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
                for elem_pair in inputs_all_prop])
            
            all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
            all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
            if all_inp_shuffled_prop:
                inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
            else:
                inputs_all_prop, targets_all_prop, nodes_all_prop = [], [], []

            batch_size = min(self.batch_size, len(inputs_all_ent))
            num_batches = int(ceil(len(inputs_all_ent)/batch_size))
            batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))

            for batch_idx in range(num_batches):
                batch_start_ent = batch_idx * batch_size
                batch_end_ent = (batch_idx+1) * batch_size
                batch_start_prop = batch_idx * batch_size_prop
                batch_end_prop = (batch_idx+1) * batch_size_prop
                
                inputs_ent = np.array(self.to_feature(inputs_all_ent[batch_start_ent: batch_end_ent], self.max_pathlen, self.max_paths, self.max_types))
                targets_ent = np.array(targets_all_ent[batch_start_ent: batch_end_ent])
                nodes_ent = np.array(nodes_all_ent[batch_start_ent: batch_end_ent])

                inputs_prop = np.array(self.pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop], max_prop_len))
                targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
                nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
                
                targets = np.concatenate((targets_ent, targets_prop), axis=0)

                inp_elems = torch.LongTensor(inputs_ent).to(self.device)
                node_elems = torch.LongTensor(nodes_ent).to(self.device)
                targ_elems = torch.DoubleTensor(targets).to(self.device)

                inp_props = torch.LongTensor(inputs_prop).to(self.device)
                node_props = torch.LongTensor(nodes_prop).to(self.device)

                optimizer.zero_grad()
                outputs = model(node_elems, inp_elems, node_props, inp_props, max_prop_len)

                loss = F.mse_loss(outputs, targ_elems)
                loss.backward()
                optimizer.step()

                if batch_idx%5000 == 0:
                    print ("Epoch: {} Idx: {} Loss: {}".format(epoch, batch_idx, loss.item()))  
                    epoch_list.append(epoch)
                    loss_list.append(loss.item())
        print ("Training complete!")

        model.eval()

        print ("Optimizing threshold...")
        self.optimize_threshold(model, self.val_data_t_ent, self.batch_size)
        
        threshold_results_mean = {el: np.mean(threshold_results[el], axis=0) for el in threshold_results}    
        threshold = max(threshold_results_mean.keys(), key=(lambda key: threshold_results_mean[key][2]))
        threshold_list.append(threshold_results)

        model.threshold = nn.Parameter(torch.DoubleTensor([threshold]))

        torch.save(model.state_dict(), self.model_path)
        print ("Done. Saved model at {}".format(self.model_path))

    def load_alignment(self):
        ontologies_in_alignment = []
        alignments = []
        for f in os.listdir(self.alignment_folder):
            doc = minidom.parse(self.alignment_folder + f)
            ls = list(zip(doc.getElementsByTagName('entity1'), doc.getElementsByTagName('entity2')))
            src = self.train_folder + doc.getElementsByTagName('Ontology')[0].getAttribute("rdf:about").split("/")[-1].rsplit(".", 1)[0] + ".owl"
            targ = self.train_folder + doc.getElementsByTagName('Ontology')[1].getAttribute("rdf:about").split("/")[-1].rsplit(".", 1)[0] + ".owl"
            ontologies_in_alignment.append((src, targ))
            alignments.extend([(a.getAttribute('rdf:resource'), b.getAttribute('rdf:resource')) for (a,b) in ls])
        
        return alignments, ontologies_in_alignment

    def split_data(self, index=0):
        # Preprocessing and parsing input data for training
        alignments, ontologies_in_alignment = self.load_alignment()
        preprocessing = GraphParser(ontologies_in_alignment, alignments)
        self.data_ent, self.data_prop, self.emb_indexer_new, self.emb_indexer_inv_new, self.emb_vals_new, self.neighbours_dicts_ent, self.neighbours_dicts_prop, self.max_types = preprocessing.process(spellcheck=True)
        emb_indexer_cached, emb_indexer_inv_cached, emb_vals_cached = {}, {}, []
        self.emb_vals, self.emb_indexer, self.emb_indexer_inv = list(emb_vals_cached), dict(emb_indexer_cached), dict(emb_indexer_inv_cached)
        s = set(self.emb_indexer.keys())
        idx = len(self.emb_indexer_inv)
        for term in self.emb_indexer_new:
            if term not in s:
                self.emb_indexer[term] = idx
                self.emb_indexer_inv[idx] = term
                self.emb_vals.append(self.emb_vals_new[self.emb_indexer_new[term]])
                idx += 1

        global DIRECT_INPUTS, DIRECT_TARGETS
        global threshold_results
        torch.set_default_dtype(torch.float64)

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        data_items = self.data_ent.items()

        np.random.shuffle(list(data_items))

        data_ent = OrderedDict(data_items)

        data_items = self.data_prop.items()
        np.random.shuffle(list(data_items))
        data_prop = OrderedDict(data_items)

        print ("Number of entity pairs:", len(data_ent))
        print ("Number of property pairs:", len(data_prop))

        torch.set_default_dtype(torch.float64)


        ontologies_in_alignment = [tuple([elem.split("/")[-1].split(".")[0] for elem in pair]) for pair in ontologies_in_alignment]
        if self.ontology_split:
            # We split on the ontology-pair level
            step = int(len(ontologies_in_alignment)/self.K)
            
            val_onto = ontologies_in_alignment[len(ontologies_in_alignment)-step+1:]
            train_data_ent = {elem: data_ent[elem] for elem in data_ent if tuple([el.split("#")[0] for el in elem]) not in val_onto}
            val_data_ent = {elem: data_ent[elem] for elem in data_ent if tuple([el.split("#")[0] for el in elem]) in val_onto}
            
            train_data_prop = {elem: data_prop[elem] for elem in data_prop if tuple([el.split("#")[0] for el in elem]) not in val_onto}
            val_data_prop = {elem: data_prop[elem] for elem in data_prop if tuple([el.split("#")[0] for el in elem]) in val_onto}
            
            self.train_data_t_ent = [key for key in train_data_ent if train_data_ent[key]]
            self.train_data_f_ent = [key for key in train_data_ent if not train_data_ent[key]]

            self.train_data_t_prop = [key for key in train_data_prop if train_data_prop[key]]
            self.train_data_f_prop = [key for key in train_data_prop if not train_data_prop[key]]

            self.val_data_t_ent = [key for key in val_data_ent if val_data_ent[key]]
            self.val_data_f_ent = [key for key in val_data_ent if not val_data_ent[key]]

            self.val_data_t_prop = [key for key in val_data_prop if val_data_prop[key]]
            self.val_data_f_prop = [key for key in val_data_prop if not val_data_prop[key]]

        else:
            # We split on the mapping-pair level
            ratio = float(1/self.K)
            data_t_ent = {elem: data_ent[elem] for elem in data_ent if data_ent[elem]}
            data_f_ent = {elem: data_ent[elem] for elem in data_ent if not data_ent[elem]}

            data_t_prop = {elem: data_prop[elem] for elem in data_prop if data_prop[elem]}
            data_f_prop = {elem: data_prop[elem] for elem in data_prop if not data_prop[elem]}

            data_t_items_ent = list(data_t_ent.keys())
            data_f_items_ent = list(data_f_ent.keys())

            data_t_items_prop = list(data_t_prop.keys())
            data_f_items_prop = list(data_f_prop.keys())

            self.val_data_t_ent = data_t_items_ent[int((ratio*index)*len(data_t_ent)):int((ratio*index + ratio)*len(data_t_ent))]
            self.val_data_f_ent = data_f_items_ent[int((ratio*index)*len(data_f_ent)):int((ratio*index + ratio)*len(data_f_ent))]

            self.train_data_t_ent = data_t_items_ent[:int(ratio*index*len(data_t_ent))] + data_t_items_ent[int(ratio*(index+1)*len(data_t_ent)):]
            self.train_data_f_ent = data_f_items_ent[:int(ratio*index*len(data_f_ent))] + data_f_items_ent[int(ratio*(index+1)*len(data_f_ent)):]

            self.val_data_t_prop = data_t_items_prop[int((ratio*index)*len(data_t_prop)):int((ratio*index + ratio)*len(data_t_prop))]
            self.val_data_f_prop = data_f_items_prop[int((ratio*index)*len(data_f_prop)):int((ratio*index + ratio)*len(data_f_prop))]

            self.train_data_t_prop = data_t_items_prop[:int(ratio*index*len(data_t_prop))] + data_t_items_prop[int(ratio*(index+1)*len(data_t_prop)):]
            self.train_data_f_prop = data_f_items_prop[:int(ratio*index*len(data_f_prop))] + data_f_items_prop[int(ratio*(index+1)*len(data_f_prop)):]

        np.random.shuffle(self.train_data_f_ent)
        self.train_data_f_ent = self.train_data_f_ent[:self.max_false_examples]

        np.random.shuffle(self.train_data_f_prop)
        self.train_data_f_prop = self.train_data_f_prop[:self.max_false_examples]

        # Oversampling to maintain 1:1 ratio between positives and negatives
        self.train_data_t_ent = np.repeat(self.train_data_t_ent, ceil(len(self.train_data_f_ent)/len(self.train_data_t_ent)), axis=0)
        self.train_data_t_ent = self.train_data_t_ent[:len(self.train_data_f_ent)].tolist()

        if self.train_data_t_prop:
            self.train_data_t_prop = np.repeat(self.train_data_t_prop, ceil(len(self.train_data_f_prop)/len(self.train_data_t_prop)), axis=0)
            self.train_data_t_prop = self.train_data_t_prop[:len(self.train_data_f_prop)].tolist()
    
    def cos_sim(self, a,b):
        return 1 - spatial.distance.cosine(a, b)
    def optimize_threshold(self, model, val_data_t_ent, batch_size):
        '''
        Function to optimise threshold on validation set.
        Calculates performance metrics (precision, recall, F1-score, F2-score, F0.5-score) for a
        range of thresholds, dictated by the range of scores output by the model, with step size 
        0.001 and updates `threshold_results` which is the relevant dictionary.
        '''

        all_results = OrderedDict()
        global DIRECT_INPUTS, DIRECT_TARGETS
        with torch.no_grad():
          
            
            np.random.shuffle(self.val_data_t_ent)
            np.random.shuffle(self.val_data_f_ent)

            np.random.shuffle(self.val_data_t_prop)
            np.random.shuffle(self.val_data_f_prop)

            # Create two sets of inputs: one for entities and one for properties

            inputs_pos, nodes_pos, targets_pos = self.generate_input(self.val_data_t_ent, 1, self.neighbours_dicts_ent,  self.emb_indexer)
            inputs_neg, nodes_neg, targets_neg = self.generate_input(self.val_data_f_ent, 0, self.neighbours_dicts_ent,  self.emb_indexer)
            inputs_pos_prop, nodes_pos_prop, targets_pos_prop = self.generate_input(self.val_data_t_prop, 1, self.neighbours_dicts_prop,  self.emb_indexer)
            inputs_neg_prop, nodes_neg_prop, targets_neg_prop = self.generate_input(self.val_data_f_prop, 0, self.neighbours_dicts_prop,  self.emb_indexer)

            inputs_all = list(inputs_pos) + list(inputs_neg)
            targets_all = list(targets_pos) + list(targets_neg)
            nodes_all = list(nodes_pos) + list(nodes_neg)
            
            all_inp = list(zip(inputs_all, targets_all, nodes_all))
            all_inp_shuffled = random.sample(all_inp, len(all_inp))
            inputs_all, targets_all, nodes_all = list(zip(*all_inp_shuffled))

            inputs_all_prop = list(inputs_pos_prop) + list(inputs_neg_prop)
            targets_all_prop = list(targets_pos_prop) + list(targets_neg_prop)
            nodes_all_prop = list(nodes_pos_prop) + list(nodes_neg_prop)
            
            all_inp_prop = list(zip(inputs_all_prop, targets_all_prop, nodes_all_prop))
            all_inp_shuffled_prop = random.sample(all_inp_prop, len(all_inp_prop))
            if all_inp_shuffled_prop:
                inputs_all_prop, targets_all_prop, nodes_all_prop = list(zip(*all_inp_shuffled_prop))
            else:
                inputs_all_prop, targets_all_prop, nodes_all_prop = [], [], []

            if len(inputs_all_prop) == 0:
                max_prop_len = 0
            else:
                max_prop_len = np.max([[[len(elem) for elem in prop] for prop in elem_pair] 
                for elem_pair in inputs_all_prop])

            batch_size = min(self.batch_size, len(inputs_all))
            num_batches = int(ceil(len(inputs_all)/batch_size))
            batch_size_prop = int(ceil(len(inputs_all_prop)/num_batches))
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = (batch_idx+1) * batch_size
                batch_start_prop = batch_idx * batch_size_prop
                batch_end_prop = (batch_idx+1) * batch_size_prop

                inputs = np.array(self.to_feature(inputs_all[batch_start: batch_end], self.max_pathlen, self.max_paths, self.max_types))
                targets = np.array(targets_all[batch_start: batch_end])
                nodes = np.array(nodes_all[batch_start: batch_end])

                inputs_prop = np.array(self.pad_prop(inputs_all_prop[batch_start_prop: batch_end_prop], max_prop_len))
                targets_prop = np.array(targets_all_prop[batch_start_prop: batch_end_prop])
                nodes_prop = np.array(nodes_all_prop[batch_start_prop: batch_end_prop])
                
                targets = np.concatenate((targets, targets_prop), axis=0)

                inp_elems = torch.LongTensor(inputs).to(self.device)
                node_elems = torch.LongTensor(nodes).to(self.device)
                targ_elems = torch.DoubleTensor(targets).to(self.device)

                inp_props = torch.LongTensor(inputs_prop).to(self.device)
                node_props = torch.LongTensor(nodes_prop).to(self.device)

                # Run model on entities and properties 
                outputs = model(node_elems, inp_elems, node_props, inp_props, max_prop_len)
                outputs = [el.item() for el in outputs]
                targets = [True if el.item() else False for el in targets]

                for idx, pred_elem in enumerate(outputs):
                    if idx < len(nodes):
                        ent1 = self.emb_indexer_inv[nodes[idx][0]]
                        ent2 = self.emb_indexer_inv[nodes[idx][1]]
                    else:
                        ent1 = self.emb_indexer_inv[nodes_prop[idx-len(nodes)][0]]
                        ent2 = self.emb_indexer_inv[nodes_prop[idx-len(nodes)][1]]
                    if (ent1, ent2) in all_results:
                        print ("Error: ", ent1, ent2, "already present")
                    all_results[(ent1, ent2)] = (pred_elem, targets[idx])
            
            DIRECT_TARGETS = [True if el else False for el in DIRECT_TARGETS]
            
            print ("Len (direct inputs): ", len(DIRECT_INPUTS))
            for idx, direct_input in enumerate(DIRECT_INPUTS):
                ent1 = self.emb_indexer_inv[direct_input[0]]
                ent2 = self.emb_indexer_inv[direct_input[1]]
                sim = self.cos_sim(self.emb_vals[direct_input[0]], self.emb_vals[direct_input[1]])
                all_results[(ent1, ent2)] = (round(sim, 3), DIRECT_TARGETS[idx])
            
            # Low threshold is lowest value output by model and high threshold is the highest value
            low_threshold = round(np.min([el[0] for el in all_results.values()]) - 0.02, 3)
            high_threshold = round(np.max([el[0] for el in all_results.values()]) + 0.02, 3)
            threshold = low_threshold
            step = 0.001

            if not self.val_data_t_prop:
                val_data_t_tot = val_data_t_ent
            else:
                val_data_t_tot = [tuple(pair) for pair in np.concatenate((val_data_t_ent, self.val_data_t_prop), axis=0)]
            # Iterate over every threshold with step size of 0.001 and calculate all evaluation metrics
            while threshold < high_threshold:
                threshold = round(threshold, 3)
                res = []
                for i,key in enumerate(all_results):
                    if all_results[key][0] > threshold:
                        res.append(key)
                s = set(res)
                fn_list = [(key, all_results[key][0]) for key in val_data_t_tot if key not in s]
                fp_list = [(elem, all_results[elem][0]) for elem in res if not all_results[elem][1]]
                tp_list = [(elem, all_results[elem][0]) for elem in res if all_results[elem][1]]
                
                tp, fn, fp = len(tp_list), len(fn_list), len(fp_list)
                
                
                try:
                    precision = tp/(tp+fp)
                    recall = tp/(tp+fn)
                    f1score = 2 * precision * recall / (precision + recall)
                    f2score = 5 * precision * recall / (4 * precision + recall)
                    f0_5score = 1.25 * precision * recall / (0.25 * precision + recall)
                except:
                    exception = True
                    step = 0.001
                    threshold += step
                    continue

                if threshold in threshold_results:
                    threshold_results[threshold].append([precision, recall, f1score, f2score, f0_5score])
                else:
                    threshold_results[threshold] = [[precision, recall, f1score, f2score, f0_5score]]
                threshold += step
    
    def is_valid(self, test_onto, key):
        return tuple([el.split("#")[0] for el in key]) not in test_onto

    def generate_data_neighbourless(self, elem_tuple, emb_indexer):
        return [emb_indexer[elem] for elem in elem_tuple]

    def embedify(self,seq, emb_indexer):
        for item in seq:
            if isinstance(item, list):
                yield list(self.embedify(item, emb_indexer))
            else:
                yield emb_indexer[item]

    def pad_prop(self,inputs, max_prop_len):
        inputs_padded = [[[elem + [0 for i in range(max_prop_len - len(elem))]
                            for elem in prop]
                        for prop in elem_pair]
                    for elem_pair in inputs]
        return inputs_padded

    def generate_data(self,elem_tuple, neighbours_dicts, emb_indexer):
        return list(self.embedify([neighbours_dicts[elem]for elem in elem_tuple], emb_indexer))

    def to_feature(self,inputs, max_pathlen, max_paths, max_types ):
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

    def generate_input(self, elems, target, neighbours_dicts, emb_indexer):
        inputs, targets, nodes = [], [], []
        global DIRECT_INPUTS, DIRECT_TARGETS
        for elem in list(elems):
            try:
                inputs.append(self.generate_data(elem, neighbours_dicts, emb_indexer))
                nodes.append(self.generate_data_neighbourless(elem, emb_indexer))
                targets.append(target)
            except KeyError as e:
                DIRECT_INPUTS.append(self.generate_data_neighbourless(elem, emb_indexer))
                DIRECT_TARGETS.append(target)
            except Exception as e:
                raise
        return inputs, nodes, targets

    def count_non_unk(self, elem):
        return len([l for l in elem if l!="<UNK>"])

if __name__ == "__main__":

    trainer = Trainer()
    trainer.split_data()
    trainer.train()

