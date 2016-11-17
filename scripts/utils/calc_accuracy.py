import json
import caffe
import glob
import os
import numpy as np
import json
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=False, help="Trained model name (atka, beacon, vgg16, etc)")
parser.add_argument("--layer_name", required=False, help="Prediction layer name (prob, prob1, prob2, etc)")
args = parser.parse_args()

model_name = args.model_name
output_layer = args.layer_name
predictions = np.load('../../predictions/event_predictions_finetune_%s_%s.npy' % (model_name, output_layer))
# predictions = np.load('../../predictions/final_predictions.npy')
top_5_event_predictions = predictions.argsort()[:,::-1][:,:5] 

test_filenames = []
test_labels = [] 
fp = open('../../models/finetune_nin/test_manifest')

for line in fp:
    test_filenames.append(line.rstrip().split()[0].split('/')[-1])
    test_labels.append(int(line.rstrip().split()[1]))
fp.close()

count_1 = 0
count_5 = 0
count_all = 0
for i in range(len(test_filenames)):
    test_label = test_labels[i]
    event_preds = top_5_event_predictions[i]
    count_all += 1
    if event_preds[0] == test_label:
        match_status = 'top-1'
        count_1 += 1
    elif test_label in event_preds:
        match_status = 'top-5'
        count_5 += 1
    else:
        match_status = 'none'

print "correct top1 test labels: %d/%d acc %f" % (count_1, count_all, count_1/float(count_all))
print "correct top5 test labels: %d/%d acc %f" % ((count_5 + count_1), count_all, (count_5 + count_1)/float(count_all))
