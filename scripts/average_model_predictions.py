import numpy as np
import glob

x = glob.glob('../predictions/event_predictions_*.npy')

count = 0
out = None
for i in x:
    y = np.load(i)
    print "Adding %s" % i
    if out is None:
        out = y
    else:
        out += y
    count += 1

final_predictions = out/float(count)
np.save('../predictions/final_predictions.npy', final_predictions)
