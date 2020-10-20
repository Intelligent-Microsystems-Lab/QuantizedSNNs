import numpy as np
import pickle
import os
import torch
import pickle


def read_aedat31(filename, labels_f, test_set = False):
    # https://inivation.com/support/software/fileformat/#aedat-31
    # http://research.ibm.com/dvsgesture/
    gestures_full = []
    labels_full = []

    # Addresses will be interpreted as 32 bits
    print(filename)
    f = open(filename, 'r', encoding='latin_1')
    labels = np.genfromtxt(labels_f, delimiter=',')[1:]
    #Skip header lines
    bof = f.tell()
    line = f.readline()
    while (line[0]=='#'):
        print(line, end='')
        bof = f.tell()
        line = f.readline()

    # read data
    f.seek(bof,0)
    dataArray = np.fromfile(f, '<u4') #little endian
    f.close()

    # extract events
    allAddr = np.array([], dtype=np.uint32)
    allTs = np.array([], dtype=np.uint32)
    pos = 0
    while pos < len(dataArray):
        num_events = dataArray[pos + 5]
        num_valid = dataArray[pos + 6]
        allAddr = np.append(allAddr, dataArray[pos+7:pos+7+(num_events*2):2])
        allTs = np.append(allTs, dataArray[pos+8:pos+8+(num_events*2):2])
        pos = pos+7+(num_events*2)

    # interpret events as x,y,polarity
    addr = allAddr
    x = ( addr >> 17 ) & 0x00001FFF
    y = ( addr >> 2 ) & 0x00001FFF
    polarity = ( addr >> 1 ) & 0x00000001

    # how to access header info
    # dataArray[0] >> 16          # event type -> polarity event
    # dataArray[0] & 0xFFFF0000   # event source ID
    # dataArray[1]                # eventSize
    # dataArray[2]                # eventTSOffset
    # dataArray[3]                # eventTSOverflow
    # dataArray[4]                # eventCapacity (always equals eventNumber)
    # dataArray[5]                # eventNumber (valid + invalid)
    # dataArray[6]                # eventValid

    stim = np.array([allTs, x, y, polarity]).T#.astype(int)

    for i in labels:

        # chop things right
        single_gesture = stim[stim[:, 0] >= i[1]]
        single_gesture = single_gesture[single_gesture[:, 0] <= i[2]]

        # bin them 1ms
        single_gesture[:,0] = np.floor(single_gesture[:,0]/1000)
        single_gesture[:,0] = single_gesture[:,0] - np.min(single_gesture[:,0])

        if test_set:
            single_gesture = single_gesture[single_gesture[:,0] <= 1800]

        if i[0] in labels_full:
            gestures_full[labels_full.index(i[0])] = np.vstack((gestures_full[labels_full.index(i[0])], single_gesture))
        else:
            gestures_full.append(single_gesture)
            # record label
            labels_full.append(i[0])
    return gestures_full, labels_full



gestures_full = []
labels_full = []
with open('trials_to_train.txt') as fp:
    for cnt, line in enumerate(fp):
        try:
            gestures_temp, labels_temp = read_aedat31(line.split(".")[0] + ".aedat", line.split(".")[0] + "_labels.csv")
            gestures_full += gestures_temp
            labels_full += labels_temp
        except:
            continue

with open('train_dvs_gesture.pickle', 'wb') as handle:
    pickle.dump((gestures_full, labels_full), handle)




gestures_full = []
labels_full = []
with open('trials_to_test.txt') as fp:
    for cnt, line in enumerate(fp):
        try:
            gestures_temp, labels_temp = read_aedat31(line.split(".")[0] + ".aedat", line.split(".")[0] + "_labels.csv", test_set = True)
            gestures_full += gestures_temp
            labels_full += labels_temp
        except:
            continue

with open('test_dvs_gesture.pickle', 'wb') as handle:
    pickle.dump((gestures_full, labels_full), handle)


