import os
import cv2
import sys
import json
import argparse
import matplotlib.pyplot as plt

IMAGES_PER_LINE = 5
MAX_LINES = 5
LABELS = ['price','main_image','name']

DOM_PATH = '../data_shops/labeled_dom_trees/'
PATCHES_PATH = '../data_shops/review_patches/'
DOWNLOADED_PAGES_PATH = '../data_shops/downloaded_pages/'
PAGE_SETS_PATH = '../data_shops/page_sets/'

PAGES_TO_DELETE = set()

# find labeled elements
def getLabeledElements(dom_path):
    results = {}

    with open(dom_path,'r') as f:
            root = json.load(f)

    processing_stack = []
    processing_stack.append(root)
    while len(processing_stack)!=0:
        node = processing_stack.pop()

        # get label
        if 'label' in node:
            label = node['label']
            results[label] = node
 
        # follow children
        if 'childNodes' in node:
            childNodes = node['childNodes']
            for childNode in childNodes:
                processing_stack.append(childNode)

    return results

def getPatch(im, element):
    pos = element['position']
    return im[pos[1]:pos[3],pos[0]:pos[2],:]

def getPageList(prefix):
    pages_path = os.path.join(DOWNLOADED_PAGES_PATH, prefix+'.txt')
    with open(pages_path,'r') as f:
        pages = [line.split('\t')[0] for line in f.readlines()]
    return pages


def preparePatches(prefix):
    # create patches directory if it does not exist
    if not os.path.exists(PATCHES_PATH):
        os.makedirs(PATCHES_PATH)

    # load pages
    pages = getPageList(prefix)


    number_of_pages = len(pages)

    # for each page from prexix
    for page_ind in range(len(pages)):
        page = pages[page_ind]

        # prepare paths
        dom_path = os.path.join(DOM_PATH, page+'.json')
        page_image_path = os.path.join('../data_shops/images/', page+'.jpeg')
        
        if os.path.isfile(dom_path):
            # get labeled elements
            labeled = getLabeledElements(dom_path)

            # load image
            im = cv2.imread(page_image_path)

            for i in range(len(LABELS)):
                label = LABELS[i]
                patch = getPatch(im,labeled[label])
                cv2.imwrite(os.path.join(PATCHES_PATH, page+'_'+label+'.jpeg'),patch)

                ### TODO lower resolution

def onPick(event):
    ax = event.inaxes
    if ax:
        page = ax.page

        if page not in PAGES_TO_DELETE:
            PAGES_TO_DELETE.add(page)

            print '*', page, 'will be removed'

            ax.spines['bottom'].set_color('red')
            ax.spines['top'].set_color('red') 
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')

            ax.spines['bottom'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
        else:
            PAGES_TO_DELETE.remove(page)

            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black') 
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')

            ax.spines['bottom'].set_linewidth(1)
            ax.spines['top'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)

            print '*', page, 'will not be removed'
        

        ax.figure.canvas.draw()


def keyPress(event):
    if event.key == 'enter':
        plt.close()

def reviewPatches(prefix):
    # load pages
    pages = getPageList(prefix)
    batch_size = IMAGES_PER_LINE*MAX_LINES


    for label in LABELS:
        print 'Please, review label:', label

        # for each batch
        for i in range(0,len(pages),batch_size):
            # get batch index
            indFrom = i
            indTo = i+batch_size

            # create figure and add subplots
            fig = plt.figure() #(figsize=(10,15))
            j = 1
            for show_page in pages[indFrom:indTo]:

                labeled_dom_path = os.path.join(DOM_PATH, show_page+'.json')

                # if we have labeled DOM -> review
                if os.path.isfile(labeled_dom_path):
                    path_to_patch = os.path.join(PATCHES_PATH,show_page+'_'+label+'.jpeg')
                    patch = cv2.imread(path_to_patch)
                    ax = fig.add_subplot(MAX_LINES,IMAGES_PER_LINE,j)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])

                    ax.page = show_page
                    ax.imshow(patch)
                # if do not have labeled DOM -> remove it
                else:
                    PAGES_TO_DELETE.add(show_page)

                j+=1

            fig.canvas.mpl_connect('key_press_event', keyPress)
            fig.canvas.mpl_connect('button_press_event', onPick)

            plt.show()

    # if result directory does not exist, create it
    if not os.path.exists(PAGE_SETS_PATH):
        os.makedirs(PAGE_SETS_PATH)


    # save result
    with open(os.path.join(PAGE_SETS_PATH,prefix+'.txt'),'w+') as f:
        for page in pages:
            if page not in PAGES_TO_DELETE:
                f.write(page+'\n')


def removePatches(prefix):
    # load pages
    pages = getPageList(prefix)

    # for each page from prexix
    for page in pages:
        # for each label
        for i in range(len(LABELS)):
            label = LABELS[i]
            patch_path = os.path.join(PATCHES_PATH, page+'_'+label+'.jpeg')

            # if it exists remove
            if os.path.isfile(patch_path):
                os.remove(patch_path)

#----- MAIN PART
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=str, choices=['prepare', 'review', 'remove'], help='phase of review process')
    parser.add_argument('prefix', type=str, help='prefix of eshop')
    args = parser.parse_args()
    
    # if phase is to prepare 
    if args.phase == 'prepare':
        preparePatches(args.prefix)

    if args.phase == 'review':
        reviewPatches(args.prefix)

    if args.phase == 'remove':
        removePatches(args.prefix)