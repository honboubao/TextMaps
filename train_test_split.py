import os
import sys
import caffe
import pickle
import random
import argparse
import numpy as np
import google.protobuf as pb2
from caffe.proto import caffe_pb2
from sklearn.preprocessing import normalize
from scipy.ndimage.filters import gaussian_filter

# DEFINE PATHS
solver_path = 'models/both_classif_solver.prototxt'
test_model = '/storage/plzen1/home/gogartom/TextMaps/models/both_classif_inference.prototxt'
pretrained_weights = '/storage/plzen1/home/gogartom/DOM-Extraction/data/imagenet_models/CaffeNet.v2.caffemodel'
split_directory = 'data_shops/page_sets/splits/'
boxes_directory = 'data_shops/input_boxes/'
snapshots_directory = 'models/training/'
test_results_directory = 'models/test_results/'
priors_directory = 'data_shops/split_priors/'


# CONSTANTS
max_x = 1280
max_y = 1280

def get_position_probabilities(position_maps, boxes):
    box_i = 0
    probs = np.zeros((boxes.shape[0],len(position_maps)),dtype=np.float32)

    for box_i in range(boxes.shape[0]):
        for cls in range(len(position_maps)):
            map = position_maps[cls]
            box = boxes[box_i]
            box_map = map[box[1]:box[3],box[0]:box[2]]
            box_cls_prob = np.mean(box_map)
            probs[box_i,cls] = box_cls_prob
    return probs

def get_results_with_position(boxes, local_probs, position_maps):
    #-- get position probability for each box
    position_probs = get_position_probabilities(position_maps, boxes)

    #-- multiply with local prob
    probs = (local_probs*position_probs)[:,1:4]
   
    #-- are the first 3 those with maximum probability
    max_inds = np.argmax(probs,axis=0)
    results = [0]*3
    for cls in range(0,3):
        if max_inds[cls] == cls:
            results[cls]=1

    return results


def test_net(snapshot_path, test_data, test_iters, position_maps):
    test_net = caffe.Net(test_model, snapshot_path, caffe.TEST)
    test_net.layers[0].set_data(test_data)

    price_results = []
    name_results = []
    image_results = []

    position_price_results = []
    position_name_results = []
    position_image_results = []

    # go through data
    for i in range(test_iters):
        test_net.forward()
        
        #--- net results
        price_results.append(test_net.blobs['web_price_accuracy'].data[0])
        image_results.append(test_net.blobs['web_image_accuracy'].data[0])
        name_results.append(test_net.blobs['web_name_accuracy'].data[0])

        #--- results with position maps
        local_probs = test_net.blobs['prob'].data[:,0:4,0,0]
        boxes = test_net.blobs['boxes'].data[:,1:5]  
        results_with_position = get_results_with_position(boxes, local_probs, position_maps)
        position_price_results.append(results_with_position[0])
        position_image_results.append(results_with_position[1])
        position_name_results.append(results_with_position[2])


    # stop fetcher
    test_net.layers[0].stop_fetcher()

    # compute net results
    image_accuracy = np.mean(image_results)
    price_accuracy = np.mean(price_results)
    name_accuracy = np.mean(name_results)
    
    # compute position results
    position_image_accuracy = np.mean(position_image_results)
    position_price_accuracy = np.mean(position_price_results)
    position_name_accuracy = np.mean(position_name_results)


    net_results = (image_accuracy, price_accuracy, name_accuracy)
    position_results = (position_image_accuracy, position_price_accuracy, position_name_accuracy)

    return net_results, position_results

def snapshot(net, snapshot_path):
    net.save(snapshot_path)

def get_snapshot_name(split_name, iter):
    return os.path.join(snapshots_directory,'snapshot_split_'+split_name+'_'+str(iter)+'.caffemodel')

def create_position_maps(train_data, split_name):
    print 'Creating and saving position maps'

    n_samples = 500

    #--- GET TRAINING DATA
    with open(train_data) as f:
        train_pages = [line.strip() for line in f.readlines()]

    #--- GET RANDOM SUBSET
    sample_pages = random.sample(train_pages, n_samples)

    #--- FROM EACH RANDOM PAGE, USE POSITION OF GT ELEMENTS 
    final_maps = [np.ones((max_y,max_x),dtype=np.float32)]*4
    for page in sample_pages:
        # load boxes
        boxes_path = os.path.join(boxes_directory, page+'.pkl')
        with open(boxes_path,'rb') as f:
            boxes = pickle.load(f)
        gt_boxes = boxes['gt_boxes']

        # add to finals
        background_map = np.ones((max_y,max_x),dtype=np.float32) # we add one (we do not want zero probability)
        for i in range(1,4):
            bb = gt_boxes[i-1]
            background_map[bb[1]:bb[3],bb[0]:bb[2]] = 0
            type_map = np.zeros((max_y,max_x),dtype=np.float32)    
            type_map[bb[1]:bb[3],bb[0]:bb[2]] = 1
            final_maps[i] = final_maps[i] + type_map

        final_maps[0] = final_maps[0] + background_map


    #--- normalize
    for i in range(0,4):
        final_maps[i] = final_maps[i]/(1+n_samples)

    #--- save
    for i in range(0,4):
        path = os.path.join(priors_directory,'split_'+split_name+'_'+str(i)+'.pkl')
        pickle.dump(final_maps[i], open(path,"wb"))

def load_position_maps(split_name, sigma):
    print 'Loading position maps smoothed with Gausian filter, sigma=',sigma

    maps = []
    for i in range(4):
        ### Load map
        path = os.path.join(priors_directory,'split_'+split_name+'_'+str(i)+'.pkl')
        map = pickle.load(open(path,'rb'))

        ### Gaussian convolution
        filtered_maps = []
        filtered =  gaussian_filter(map, sigma)
        maps.append(filtered)

    return maps


#----- MAIN PART
if __name__ == "__main__":

    #--- Get params
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, help='Split number')
    parser.add_argument('--train_iters', type=int, help='Number of iterations for training')
    parser.add_argument('--test_iters', type=int, help='Number of iterations for testing')
    args = parser.parse_args()
    
    #-- Load params
    split_name = str(args.split)
    train_iters = args.train_iters
    final_test_iters = args.test_iters

    #--- GET DATA PATHS
    train_data = os.path.join(split_directory,'split_'+split_name+'_train.txt')
    test_data = os.path.join(split_directory,'split_'+split_name+'_test.txt')

    #--- CREATE POSITION MAPS
    create_position_maps(train_data, split_name)

    #--- LOAD SMOTHED POSITION MAPS
    position_maps = load_position_maps(split_name, 80)    

    #--- GET TEST RESULTS PATH
    test_res_path = os.path.join(test_results_directory, 'results_split_'+split_name+'.txt')
    
    ###--- LOAD SOLVER PARAMS
    solver_param = caffe_pb2.SolverParameter()
    with open(solver_path, 'rt') as f:
        pb2.text_format.Merge(f.read(), solver_param)

    # print solver_param
    ###--- LOAD SOLVER
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_path)

    ###--- TRAIN
    #-- set input data and pretrained model
    train_net = solver.net
    train_net.copy_from(pretrained_weights)
    train_net.layers[0].set_data(train_data)

    #-- make steps
    num_steps = 20
    test_every_n_iters = 200
    val_iters = 100
    for i in range(train_iters/num_steps):
        #-- update
        solver.step(num_steps)
        iteration=(i+1)*num_steps

        #-- print loss
        print '----------------------'
        print 'Iteration:',str(iteration)
        print 'Loss:',train_net.blobs['loss'].data
        sys.stdout.flush()

        #-- create snapshot and test
        if iteration%test_every_n_iters==0:
            print 'Snapshoting and Testing'
            sys.stdout.flush()

            #-- snapshot
            snapshot_path = get_snapshot_name(split_name,iteration)
            snapshot(train_net, snapshot_path)

            #-- test
            net_results, position_results = test_net(snapshot_path, test_data, val_iters, position_maps)
           
            im_acc, price_acc, name_acc = net_results
            print 'NET: image accuracy:', im_acc
            print 'NET: price accuracy:', price_acc
            print 'NET: name accuracy:', name_acc
           
            p_im_acc, p_price_acc, p_name_acc = position_results
            print 'NET+POSITION: image accuracy:', p_im_acc
            print 'NET+POSITION: price accuracy:', p_price_acc
            print 'NET+POSITION: name accuracy:', p_name_acc
            
            sys.stdout.flush()

    ###--- FINAL SNAPSHOT
    snapshot_path = get_snapshot_name(split_name,train_iters)
    train_net.save(snapshot_path)

    ###--- FINAL TEST
    print 'Final testing'
    sys.stdout.flush()

    net_results, position_results = test_net(snapshot_path, test_data, final_test_iters, position_maps)

    im_acc, price_acc, name_acc = net_results
    print 'NET: image accuracy:', im_acc
    print 'NET: price accuracy:', price_acc
    print 'NET: name accuracy:', name_acc

    p_im_acc, p_price_acc, p_name_acc = position_results
    print 'NET+POSITION: image accuracy:', p_im_acc
    print 'NET+POSITION: price accuracy:', p_price_acc
    print 'NET+POSITION: name accuracy:', p_name_acc
    sys.stdout.flush()

    ###--- save results
    with open(test_res_path, 'w+') as f:
        f.write('NET: image accuracy: '+str(im_acc)+"\n")
        f.write('NET: price accuracy: '+str(price_acc)+"\n")
        f.write('NET: name accuracy: '+str(name_acc)+"\n")
        f.write('\n')
        f.write('NET+POSITION: image accuracy: '+str(p_im_acc)+"\n")
        f.write('NET+POSITION: price accuracy: '+str(p_price_acc)+"\n")
        f.write('NET+POSITION: name accuracy: '+str(p_name_acc)+"\n")
