import os
import pickle
import argparse
import numpy as np
from dom_tree import DOMTree

LABELED_DOM_PATH = '../data_shops/labeled_dom_trees/'
PAGE_SETS_PATH = '../data_shops/page_sets/'
BOXES_PATH = '../data_shops/input_boxes/'

### labeled boxes
label_to_ind = {
    'price' : 0,
    'main_image' : 1,
    'name' : 2        
}

#----- MAIN PART
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='prefix of eshop')
    args = parser.parse_args()

    # create boxes directory if it does not exist
    if not os.path.exists(BOXES_PATH):
        os.makedirs(BOXES_PATH)

    # load pages from final page set
    page_set_path = os.path.join(PAGE_SETS_PATH, args.prefix+'.txt')
    with open(page_set_path,'r') as f:
        pages = [line.strip() for line in f.readlines()]

    # for each page
    for page in pages:
        print page

        # init boxes
        gt_boxes = np.zeros((len(label_to_ind),4),dtype = np.float32)
        other_boxes = []

        # load labeled dom tree
        dom_path = os.path.join(LABELED_DOM_PATH,page+'.json')
        dom = DOMTree(dom_path)
        
        # get leaf nodes
        leafNodes = dom.getPositionedLeafNodes()

        # for each leaf node
        for node in leafNodes:
            if 'label' in node:
                label = node['label']
                ind = label_to_ind[label]
                position = node['position']
                gt_boxes[ind,:] = position
            else:
                other_boxes.append(node['position'])

        all_boxes =  np.array(other_boxes,dtype = np.float32)

        # SAVE
        obj = {}
        obj['all_boxes'] = all_boxes 
        obj['gt_boxes'] = gt_boxes

        box = os.path.join(BOXES_PATH,page+'.pkl')
        with open(box,'wb+') as f:    
            pickle.dump(obj, f)