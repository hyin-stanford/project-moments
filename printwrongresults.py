from opts import parse_opts
import os
import json

opt= parse_opts()
opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
print (opt.annotation_path)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

data= load_annotation_data(opt.annotation_path)
class_to_idx = get_class_labels(data)

idx_to_class = {}
for name, label in class_to_idx.items():
   	idx_to_class[label] = name


def print_prediction(idx):
	print ([idx_to_class[i[0]] for i in idx])

idx= [[188], [101], [69], [129], [43]]
print_prediction(idx)

