import numpy as np
import os

prediction_name = 'final_predictions'
output_dir = '../submissions/' + prediction_name+ '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

event_preds = np.load('../predictions/%s.npy' % prediction_name)

fp = open('test_images')
test_filenames = []
for line in fp:
    test_filenames.append(line.strip())
fp.close()

fp = open('classes')
event_list = []
for line in fp:
    event_list.append(line.split()[0].split('_train')[0])
fp.close()

for i in range(len(event_list)):
    print "writing predictions for %s" % event_list[i]
    with open(output_dir+event_list[i]+'.txt', 'w') as f:
        out = zip(test_filenames, event_preds[:,i])
        for i in out:
            f.write(' '.join(str(s) for s in i) + '\n')
    f.close()
