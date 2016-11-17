import json
import caffe
import glob
import os
import numpy as np
import json
import time
import argparse


caffe.set_mode_gpu()
caffe.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Trained model name (atka, beacon, vgg16, etc)")
parser.add_argument("--layer_name", required=True, help="Prediction layer name (prob, prob1, prob2, etc)")
parser.add_argument("--mean", required=True, help="npy or bp (for vgg based models)")
parser.add_argument("--output_size", required=True, help="4096 (for vgg based models)")
args = parser.parse_args()

model_name = args.model_name
output_layer = args.layer_name
output_size = int(args.output_size)
BASE_PATH = "../../models/finetune_%s" % model_name


model_path = os.path.join(BASE_PATH, "model.caffemodel")
deploy_path = os.path.join(BASE_PATH, "deploy_layers.prototxt")

if args.mean == 'npy':
    mean = np.load(os.path.join(BASE_PATH, "mean.npy"))
elif args.mean == 'bp':
    proto_data = open(os.path.join(BASE_PATH, "VGG_mean.binaryproto"), "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]
else:
    mean = np.array([104, 117, 123]) 

net = caffe.Classifier(deploy_path, model_path,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

def generate_predictions(input_imgs):
    input_imgs = caffe.io.oversample(input_imgs, net.crop_dims)
    input_ = np.array(input_imgs)

    # if no oversampling
    """
    center = np.array(net.image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -net.crop_dims / 2.0,
        net.crop_dims / 2.0
    ])
    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
    """

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                           dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = net.transformer.preprocess(net.inputs[0], in_)

    out = net.forward_all(blobs=[output_layer], **{net.inputs[0]: caffe_in})
    predictions = out[output_layer]

    predictions = predictions.reshape((len(predictions) / 10, 10, -1))
    predictions = predictions.mean(1)
    return predictions
    
test_filenames = []
fp = open(os.path.join(BASE_PATH, 'test_manifest'))
for line in fp:
    test_filenames.append('../' + line.rstrip().split()[0][3:])
fp.close()

final_preds = np.lib.format.open_memmap('../../predictions/event_predictions_finetune_'+model_name+'_'+output_layer+'.npy', "w+", 'float16', shape=(len(test_filenames), output_size))

chunk_size = 0
inputs = []
for i in range(len(test_filenames)):
    inputs.append(caffe.io.resize_image(caffe.io.load_image(test_filenames[i]), net.image_dims))

    if (i and i % 100 == 0) or i == len(test_filenames) - 1:
        t1 = time.time()
        fc = generate_predictions(inputs)
        print "completed %d predictions in %f time" % (i, time.time() - t1)
        final_preds[chunk_size:chunk_size+len(inputs)] = fc
        chunk_size += len(inputs)
        inputs = []
