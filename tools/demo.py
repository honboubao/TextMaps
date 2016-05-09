import os
import cv2
import caffe
import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import load_position_map
from test import  get_probabilities_with_position
from custom_layers.dom_tree import DOMTree
import custom_layers.web_data_utils as data_utils

#----------------
# PREPARE PHANTOMJS DOWNLOAD PAGE (URL, TMPFILENAME) -> IMAGE.JPEG, DOM-TREE.JSON
# CALL PHANTOMJS WITH FILENAME
#-----------------

# TODO - predict with position -> move from test to TOOLS utils
# TODO - predict without position -> move from test to TOOLS utils
# use utils.load_position_maps
# move get_position_probabilities
# move get_probabilities_with_position
# ???? move get_results_with_position ?????

# NETWORK SETTINGS
N_FEATURES = 128
Y_SIZE = 1280
X_SIZE = 1280
SPATIAL_SHAPE = (Y_SIZE, X_SIZE)
TEXT_MAP_SCALE = 0.125
GAUSS_VAR = 80

# DATA
image_path = '../../textmaps_download/www.final-score.com-000001.jpeg'
dom_path = '../../textmaps_download/www.final-score.com-000001.json'
position_map_path = '../../textmaps_download/split_priors/'

#--- LOAD SMOTHED POSITION MAPS
position_maps = []
for i in range(4):
    path = os.path.join(position_map_path,'split_1_'+str(i)+'.pkl')
    position_maps.append(load_position_map(path,sigma=80))

# load image
im = cv2.imread(image_path)
size_x = min(im.shape[1],X_SIZE)
size_y = min(im.shape[0],Y_SIZE)

# Crop
im_croped = np.zeros((Y_SIZE,X_SIZE,3),dtype=np.uint8)
im_croped[:size_y,:size_x,:] = im[:size_y,:size_x,:] 

# TODO - this part is tensor_list_to_blob -> move to CUSTOM LAYER utils
n_channels = im.shape[2]
im_blob = np.zeros((1, Y_SIZE, X_SIZE, n_channels), dtype=np.float32)
im_blob[0, 0:im_croped.shape[0], 0:im_croped.shape[1], :] = im_croped
im_blob = im_blob.transpose((0, 3, 1, 2))

# load dom tree
dom = DOMTree(dom_path)

# get leaf nodes
leaf_nodes = dom.getPositionedLeafNodes()

# get text nodes
text_nodes = data_utils.get_text_nodes(leaf_nodes,N_FEATURES)

# get text maps
text_maps = data_utils.get_text_maps(text_nodes, N_FEATURES, SPATIAL_SHAPE, TEXT_MAP_SCALE)

# TODO - this part is tensor_list_to_blob -> move to CUSTOM LAYER utils
n_channels = text_maps.shape[2]
text_blob = np.zeros((1, text_maps.shape[0], text_maps.shape[1], n_channels), dtype=np.float32)
text_blob[0, 0:text_maps.shape[0], 0:text_maps.shape[1], :] = text_maps
text_blob = text_blob.transpose((0, 3, 1, 2))

# get input boxes
boxes = np.array([leaf['position'] for leaf in leaf_nodes],dtype = np.float32)
# remove boxes outside the considered area
keep_indices = np.logical_and.reduce(((boxes[:,0]>=0), (boxes[:,1]>=0),(boxes[:,2]<=size_x), (boxes[:,3]<=size_y)))
boxes = boxes[keep_indices,:]

boxes_this_image = np.hstack((np.zeros((boxes.shape[0], 1)), boxes))

#LOAD NET
test_model = '../../textmaps_download/v4_test.prototxt'
snapshot_path = '../../textmaps_download/snapshot_split_1_10000.caffemodel'
net = caffe.Net(test_model, snapshot_path, caffe.TEST)

# CHECK
print '--'*20
print im_blob.shape
print text_blob.shape
print boxes_this_image.shape

# SET DATA
net.blobs['im_data'].reshape(*(im_blob.shape))
net.blobs['txt_data'].reshape(*(text_blob.shape))
net.blobs['boxes'].reshape(*(boxes_this_image.shape))

# NET FORWARD
blobs_out = net.forward(im_data=im_blob.astype(np.float32, copy=False),txt_data=text_blob.astype(np.float32, copy=False),
                        boxes=boxes_this_image.astype(np.float32, copy=False))

# VISUALIZE
colors = ['r','g','b']

image = net.blobs['im_data'].data
image = np.squeeze(image[0,:,:,:])
image = image/255.0
image = np.transpose(image, (1,2,0))
image = image[:,:,(2,1,0)]
plt.imshow(image)

predicted = net.blobs['prob'].data[:,0:4,0,0]
boxes = net.blobs['boxes'].data[:,1:5] 

probs = get_probabilities_with_position(boxes, predicted, position_maps)
box_class = np.argmax(probs,axis=1)
max_boxes = np.argmax(probs,axis=0)

# print box_class
print boxes[0:2,:]

for cls in range(1,4):
    ind = max_boxes[cls]
    pred_box = boxes[ind,:]
    rect = plt.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
        pred_box[3] - pred_box[1], fill=True, alpha=0.5,facecolor=colors[cls-1],
        edgecolor=colors[cls-1], linewidth=3)
    plt.gca().add_patch(rect)

plt.show()

