import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-dir", "--dir", type = str, help = "activation bits")
args = vars(ap.parse_args())


print("hello")

with open(args['dir'] + '/test.pkl', 'wb') as f:
    pickle.dump({"foo":"bar"}, f)