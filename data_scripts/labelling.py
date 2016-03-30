import os
import cv2
import sys
import json
import copy
from dom_tree import DOMTree
import matplotlib.pyplot as plt

#----- CLASS FOR SELECTING A NODE
class ElementSelector:

    selected_patch = None

    def __init__(self, image_path, dom_tree):
        self.image_path = image_path
        self.dom_tree = dom_tree

    def onPick(self, event):
        # change back old selected page
        if self.selected_patch:
            self.selected_patch.set_linewidth(1)

        # new selected patch
        self.selected_patch = event.artist

        # draw
        self.selected_patch.set_linewidth(4)
        self.selected_patch.figure.canvas.draw()

    def keyPress(self, event):
        if event.key == 'enter':
            plt.close()

    # def getPositionedLeafNodes(self):
    #     ## FIND LEAVES WITH POSITION
    #     processing_stack = []
    #     res = []

    #     processing_stack.append(self.dom_tree.root)    
    #     while len(processing_stack)!=0:
    #         node = processing_stack.pop()

    #         # if it has children follow them
    #         if 'childNodes' in node:
    #             for childNode in node['prunedChildNodes']:
    #                 processing_stack.append(childNode)
    #         # if we have not children and element has non zero position
    #         else:
    #             if 'position' in node and ((node['position'][2]-node['position'][0])*(node['position'][3]-node['position'][1]) != 0):
    #                 res.append(node)
    #     return res

    def selectElement(self):
        ## CROP IMAGE
        crop_top = 900
        self.fig = plt.figure(figsize=(10,8))
        im = cv2.imread(self.image_path)
        im = im[:crop_top,:,:]

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        patches = []


        # for each leaf node
        for leafNode in self.dom_tree.getPositionedLeafNodes():
            position = leafNode['position']

            # text nodes have different color (just for sanity checks)
            if leafNode['type']==3:
                patch = plt.Rectangle((position[0], position[1]), position[2]-position[0],position[3]-position[1], fill=False, edgecolor='g', linewidth=1, picker=5)
            else:    
                patch = plt.Rectangle((position[0], position[1]), position[2]-position[0],position[3]-position[1], fill=False, edgecolor='b', linewidth=1, picker=5)
            
            patch.node = leafNode
            
            # compute size
            size = (position[2]-position[0])*(position[3]-position[1])
            # add to patch list
            patches.append((patch,size))

        patches.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
        for (patch,size) in patches:
            plt.gca().add_patch(patch)

        self.fig.canvas.mpl_connect('pick_event', self.onPick)
        self.fig.canvas.mpl_connect('key_press_event', self.keyPress)
        plt.show()

        if self.selected_patch:
            return self.selected_patch.node
        else:
            return None


def findAndLabel(dom, image_path, labelName, paths):
    # find element
    element = dom.getElementByOneOfPaths(paths)
    # if we have no element -> select it
    if element is None:
        print '\''+labelName+'\' not found. Please give me a hint:'
        selector = ElementSelector(image_path,dom)
        element = selector.selectElement()
    
    # if we selected the element
    if element:
        new_paths = dom.getPaths(element)
        paths.extend(new_paths)
        element['label'] = labelName
        print '\''+labelName+'\' found'
        return True

    # we still do not have element
    else:
        return False

#----- MAIN PART
if __name__ == "__main__":

    # read params
    if len(sys.argv) != 2:
        print 'BAD PARAMS. USAGE [prefix]'
        sys.exit(1)
    
    prefix = sys.argv[1]

    # prepare output path
    labeled_doms_path =  '../data_shops/labeled_dom_trees/'
    if not os.path.exists(labeled_doms_path):
        os.makedirs(labeled_doms_path)

    # load pages
    pages_path = os.path.join('../data_shops/downloaded_pages/', prefix+'.txt')
    with open(pages_path,'r') as f:
        pages = [line.split('\t')[0] for line in f.readlines()]

    # prepare paths to elements
    price_paths = []
    main_image_paths = []
    name_paths = []

    # for every page
    for page in pages:
        print '-'*20
        print page

        # prepare paths
        dom_path = os.path.join('../data_shops/dom_trees/',page+'.json')
        page_image_path = os.path.join('../data_shops/images/',page+'.jpeg')
        
        # load dom tree
        dom = DOMTree(dom_path)
        
        # get price
        price_succes = findAndLabel(dom, page_image_path, 'price', price_paths)
        
        # if we got price, get main image
        if price_succes:
            main_image_succes = findAndLabel(dom, page_image_path, 'main_image', main_image_paths)
        
        # if we got both, try to get name
        if price_succes and main_image_succes:
            name_succes = findAndLabel(dom, page_image_path, 'name', name_paths)


        if price_succes and main_image_succes and name_succes:
            dom.saveTree(os.path.join(labeled_doms_path, page+'.json'))
