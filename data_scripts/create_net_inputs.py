import re
import os
import cv2
import pickle
import argparse
import numpy as np
from dom_tree import DOMTree
from sklearn.feature_extraction.text import HashingVectorizer

IMAGES_PATH = '../data_shops/images/'
LABELED_DOM_PATH = '../data_shops/labeled_dom_trees/'
PAGE_SETS_PATH = '../data_shops/page_sets/'
BOXES_PATH = '../data_shops/input_boxes/'
TEXT_MAPS_PATH = '../data_shops/text_maps/'

### CONSTANTS
N_FEATURES = 128

### labeled boxes
label_to_ind = {
    'price' : 0,
    'main_image' : 1,
    'name' : 2        
}

### DEFINITIONS
def preprocess_string(text):

    res = re.sub("[!.;?^*()_{}|]","", text) # remove special characters (we keep characters such as "$" and ",-" )
    res = re.sub("\d+", " ^number^ ", res)   # replace numbers wit special word and space
    return res

# just split on whitespaces
def my_tokenizer(s):
    return s.split()

#----- MAIN PART
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='prefix of eshop')
    args = parser.parse_args()

    # vectorizer
    vectorizer = HashingVectorizer(n_features=N_FEATURES, tokenizer=my_tokenizer, non_negative=True, preprocessor=preprocess_string, norm=None) 

    # create boxes directory if it does not exist
    if not os.path.exists(BOXES_PATH):
        os.makedirs(BOXES_PATH)

    # create text maps directory if it does not exist
    if not os.path.exists(TEXT_MAPS_PATH):
        os.makedirs(TEXT_MAPS_PATH)

    # load pages from final page set
    page_set_path = os.path.join(PAGE_SETS_PATH, args.prefix+'.txt')
    with open(page_set_path,'r') as f:
        pages = [line.strip() for line in f.readlines()]



    # for each page
    for page in pages:
        print page

        # init boxes and text nodes
        gt_boxes = np.zeros((len(label_to_ind),4),dtype = np.float32)
        other_boxes = []
        text_nodes = []

        # load image
        image_path = os.path.join(IMAGES_PATH,page+'.jpeg')
        im = cv2.imread(image_path)
        shape = (im.shape[0],im.shape[1],N_FEATURES)

        # load labeled dom tree
        dom_path = os.path.join(LABELED_DOM_PATH,page+'.json')
        dom = DOMTree(dom_path)
        
        # get leaf nodes
        leafNodes = dom.getPositionedLeafNodes()

        # for each leaf node
        for node in leafNodes:
            #-- process input boxes
            if 'label' in node:
                label = node['label']
                ind = label_to_ind[label]
                position = node['position']
                gt_boxes[ind,:] = position
            else:
                other_boxes.append(node['position'])

            #-- process text nodes
            # if it is text node with value
            if node['type'] == 3 and 'value' in node:
                position = node['position']
                size = [(position[2]-position[0])*(position[3]-position[1])]
                
                # get text - remove whitespaces, lowercase
                text = node['value']
                text = ' '.join(text.lower().split())
                encoded_text = vectorizer.transform([text])

                if len(encoded_text.nonzero()[0]) > 0:
                    text_nodes.append((position,encoded_text,size))

        # ORDER TEXT NODES BY SIZE
        text_nodes.sort(key=lambda x: x[2], reverse=True)  

        # ALL BOXES TO NUMPY ARRAY
        other_boxes =  np.array(other_boxes,dtype = np.float32)

        #-- SAVE BOXES
        box_obj = {}
        box_obj['other_boxes'] = all_boxes 
        box_obj['gt_boxes'] = gt_boxes
        box_path = os.path.join(BOXES_PATH,page+'.pkl')
        with open(box_path,'wb+') as f:    
            pickle.dump(box_obj, f)

        #-- SAVE TEXT NODES
        txt_obj = {}
        txt_obj['shape'] = shape
        txt_obj['text_nodes'] = text_nodes
        txt_path = os.path.join(TEXT_MAPS_PATH,page+'.pkl')
        with open(txt_path,'wb+') as f:    
            pickle.dump(txt_obj, f)
