import os
import cv2
import sys
import json
import copy
#from enum import Enum
import matplotlib.pyplot as plt

#----- CLASS REPRESENTING PATH TO ELEMENT (IN DOM TREE)
class ElementPath:

    def __init__(self, root_type, root=None, localNames=None):
        self.root_type = root_type  # whether use root of tree, or unique class, or unique id
        self.root = root            # root defines where path starts
        if localNames is None:
            localNames = []
        self.localNames = localNames
    
    def __str__(self):
        if self.root_type!=ElementPath.RootType._dom_root:
            s = '['+str(self.root_type)+':'+self.root+']'
        else:
            s = '['+str(self.root_type)+']'
        for localName in self.localNames:
            s = s+'/'+localName
        return s


    class RootType:
        _dom_root = 1
        _id = 2
        _class = 3

#----- CLASS REPRESENTING DOM TREE
class DOMTree:
    def __init__(self, path):
        #-- LOAD DATA FROM FILE
        with open(path,'r') as f:
            root = json.load(f)
        
        #-- INIT VARIABLES
        self.root = root

        #-- Get auxiliary info: parent nodes, local name, get pruned version of tree
        self.addAdditionalInfo()
        self.getPrunedVersion()

    def saveTree(self, path):
        copied_root = copy.deepcopy(self.root)

        # auxiliary fields shall be removed
        fields_to_remove = ['parent', 'localName', 'prunedChildNodes', 'visibleBox']

        processing_stack = []
        processing_stack.append(copied_root)
        while len(processing_stack)!=0:
            node = processing_stack.pop()

            # remove
            for field in fields_to_remove:
                if field in node:
                    del node[field]

            # follow children
            if 'childNodes' in node:
                childNodes = node['childNodes']
                for childNode in childNodes:
                    processing_stack.append(childNode)


        print 'Saving tree to ', path
        with open(path, 'w+') as f:
            json.dump(copied_root, f, separators=(',', ':'))


    def getLocalNodeName(self, name, count):
        return name+'['+str(count)+']'

    def getPaths(self, node):
        paths = []

        ## if it has some identifier, i.e. unique id or class
        if 'attrs' in node:
            attrs = node['attrs']

            # process IDS
            if 'id' in attrs:
                for _id in attrs['id'].split():
                    # if id is in uniques
                    if _id in self.unique_ids:
                        paths.append(ElementPath(ElementPath.RootType._id, root=_id))
                # return paths          

            # process classes
            if 'class' in attrs:
                for _cls in attrs['class'].split():
                    if _cls in self.unique_classes:
                        paths.append(ElementPath(ElementPath.RootType._class, root=_cls))
        
        ## if it has not parent -> it is a root
        if 'parent' not in node:
            paths.append(ElementPath(ElementPath.RootType._dom_root))
        
        ## if it has
        else:
            # get paths to parent
            parent_paths = self.getPaths(node['parent'])
            # append local name to parent path
            for ppath in parent_paths:
                ppath.localNames.append(node['localName'])
                paths.append(ppath)      

        return paths

    def getElementByOneOfPaths(self, paths):
        node = None
        for path in paths:
            node = self.getElementByPath(path)
            
            if node is not None:
                return node
        return node


    def getElementByPath(self, path):
        # get root node, based on the type
        if path.root_type == ElementPath.RootType._dom_root:
            node = self.root
        elif path.root_type == ElementPath.RootType._class:
            if path.root in self.unique_classes:
                node = self.unique_classes[path.root]
            else:
                return None
        elif path.root_type == ElementPath.RootType._id:
            if path.root in self.unique_ids:
                node = self.unique_ids[path.root]
            else:
                return None

        # follow localNames
        for localName in path.localNames:

            # if it has no children -> path does not exist
            if 'childNodes' not in node:
                return None
            # process children
            else:
                childNamesDict = {}
                foundMatch = False
                # for every child
                for childNode in node['childNodes']:
                    # if child local name matches with our local name -> follow it
                    if localName == childNode['localName']:
                        node = childNode
                        foundMatch = True
                        break

                # if there was no match -> path does not exist
                if not foundMatch:
                    return None

        # last node is a result
        return node

    def addAdditionalInfo(self):
        self.unique_ids = {}
        self.unique_classes = {} 

        #-- GO THROUGH DATA AND SAVE SOME DATA
        # initialize processing stack with root
        processing_stack = []
        processing_stack.append(self.root)
        classes_to_delete = set()
        ids_to_delete = set()

        while len(processing_stack)!=0:
            node = processing_stack.pop()

            ###------ IDS and Classes

            # if node has some attributes
            if 'attrs' in node:
                attrs = node['attrs']
                # process IDS
                if 'id' in attrs:
                    for _id in attrs['id'].split():
                        # if id is not in uniques
                        if _id not in self.unique_ids:
                            self.unique_ids[_id] = node
                        # if it is there, its not unique -> remove
                        else:
                            ids_to_delete.add(_id)
                            
                # process classes
                if 'class' in attrs:
                    for _cls in attrs['class'].split():
                        # if class is not in uniques
                        if _cls not in self.unique_classes:
                            self.unique_classes[_cls] = node
                        # if it is there, its not unique -> remove
                        else:
                            classes_to_delete.add(_cls)
                            # del self.unique_classes[_cls]

            ###------ Children processing

            # if node has children, process and follow them
            if 'childNodes' in node:
                childNodes = node['childNodes']
                childNamesDict = {} # init child names

                for childNode in childNodes:
                    # add parent link
                    childNode['parent'] = node

                    # add local name
                    childName = childNode['name']
                    if childName in childNamesDict:
                        count = childNamesDict[childName] + 1
                    else:
                        count = 1
                    childNamesDict[childName] = count
                    localName = self.getLocalNodeName(childName, count)
                    childNode['localName'] = localName

                    # add to stack for further processing
                    processing_stack.append(childNode)

        #-- REMOVE NON-UNIQUE CLASSES AND IDS
        for _cls in classes_to_delete:
            del self.unique_classes[_cls]

        for _id in ids_to_delete:
            del self.unique_ids[_id]

        #-- PRINT SIZES OF UNIQUES
        # print '# of uniques ids', len(self.unique_ids)
        # print '# of uniques classes', len(self.unique_classes)

    # GETS PRUNED VERSION OF THE TREE        
    def getPrunedVersion(self):
        ### REMOVE EMPTY TEXT NODES
        ### REMOVE ELEMENTS WITH ZERO SIZE (AND WHOLE SUBTREE)
        ### REMOVE ELEMENTS WHICH ARE HIDDEN (AND WHOLE SUBTREE)

        processing_stack = []
        processing_stack.append(self.root)    

        while len(processing_stack)!=0:
            node = processing_stack.pop()

            if 'childNodes' in node:
                childNodes = node['childNodes']
                prunedChildNodes = []

                for childNode in childNodes:
                    ### REMOVE EMPTY TEXT NODES
                    if (childNode['type']==3) and ('value' in childNode) and (not childNode['value'].strip()):
                        continue

                    ### REMOVE ELEMENTS NON TEXT ELEMENTS WHICH DO NOT HAVE COMPUTED STYLE
                    if (childNode['type']!=3) and ('computed_style' not in childNode): 
                        continue

                    ### REMOVE HIDDEN ELEMENTS 
                    if ('computed_style' in childNode):
                        if (childNode['computed_style']['display'] == "none" or
                            childNode['computed_style']['visibility'] == "hidden" or
                            childNode['computed_style']['opacity'] == "0"):
                            continue

                    ### ADD TO CHILDREN AND TO PROCESSING STACK
                    prunedChildNodes.append(childNode)
                    processing_stack.append(childNode)

                ### SET NEW CHILDREN
                node['prunedChildNodes'] = prunedChildNodes

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

    def getPositionedLeafNodes(self):
        ## FIND LEAVES WITH POSITION
        processing_stack = []
        res = []

        processing_stack.append(self.dom_tree.root)    
        while len(processing_stack)!=0:
            node = processing_stack.pop()

            # if it has children follow them
            if 'childNodes' in node:
                for childNode in node['prunedChildNodes']:
                    processing_stack.append(childNode)
            # if we have not children and element has non zero position
            else:
                if 'position' in node and ((node['position'][2]-node['position'][0])*(node['position'][3]-node['position'][1]) != 0):
                    res.append(node)
        return res

    def selectElement(self):
        ## CROP IMAGE
        crop_top = 900
        self.fig = plt.figure(figsize=(10,8))
        im = cv2.imread(self.image_path)
        im = im[:crop_top,:,:]

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        patches = []
        # for each leaf node
        for leafNode in self.getPositionedLeafNodes():
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
