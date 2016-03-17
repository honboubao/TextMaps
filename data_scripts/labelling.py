import json
from enum import Enum

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


    class RootType(Enum):
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
        print '# of uniques ids', len(self.unique_ids)
        print '# of uniques classes', len(self.unique_classes)

        #-- TODO
        # SAVE REFERENCES TO PARENTS WITH NAMES

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
                        if _cls=='alza':
                            print 'alza in uniques:('
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

    def getElementByPath(self, path):
        # get root node, based on the type
        if path.root_type == ElementPath.RootType._dom_root:
            node = self.root
        elif path.root_type == ElementPath.RootType._class:
            node = self.unique_classes[path.root]
        elif path.root_type == ElementPath.RootType._id:
            node = self.unique_ids[path.root]

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


#----- MAIN PART
if __name__ == "__main__":
    test_path = '../data_shops/dom_trees/alza-000002.json'
    dom = DOMTree(test_path)

    path = ElementPath(ElementPath.RootType._id, root='tcb18852652', localNames=['#text[1]'])
    node = dom.getElementByPath(path)

    print 'node name', node['name']

    paths = dom.getPaths(node)
    print  'number of paths', len(paths)

    for path in paths:
        print len(path.localNames)
        node2 = dom.getElementByPath(path)
        print path
        # print node2

        print 'equals:', node==node2

