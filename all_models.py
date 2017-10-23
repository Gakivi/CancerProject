import sys
if sys.argv[1] in ['-h', '--help']:
    print('usage: sript.py feature_flags optional:combine_activities')
    print('co2 ox x y z flow hr')
    print('co2 ox accel_mag flow hr')
    quit()

import os
import mldatautil as ml

feature_flags = sys.argv[1]
if len(sys.argv) > 2:
    combine = [sys.argv[2]]
else:
    combine = ['t', 'f']

scripts = ["knn.py", "naive_bayes.py", "random_forest.py", "svm.py", "multilayer.py"]

for c in combine:
    ml.printparamsused(feature_flags, c)
    for script in scripts:
        os.system("python {} {} {}".format(script, feature_flags, c))
