import numpy as np
import pickle
import os
import torch
import pickle


def read_aedat31(filename, labels_f, test_set = False):
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
    stim[stim[:, 3] == 0, 3] = -1

    for i in labels:
        # record label
        labels_full.append(i[0])

        # chop things right
        single_gesture = stim[stim[:, 0] >= i[1]]
        single_gesture = single_gesture[single_gesture[:, 0] <= i[2]]

        # bin them 1ms
        single_gesture[:,0] = np.floor(single_gesture[:,0]/1000)
        single_gesture[:,0] = single_gesture[:,0] - np.min(single_gesture[:,0])

        if test_set:
            single_gesture = single_gesture[single_gesture[:,0] <= 1800]

        # to matrix
        #sparse_matrix = torch.sparse.FloatTensor(torch.LongTensor(single_gesture[:,[True, True, True, False]].T), torch.FloatTensor(single_gesture[:,3])).to_dense()

        # quick trick...
        #sparse_matrix[sparse_matrix < 0] = -1
        #sparse_matrix[sparse_matrix > 0] = 1

        gestures_full.append(single_gesture)
    return gestures_full, labels_full


gestures_full = []
labels_full = []
with open('trials_to_train.txt') as fp:
    for cnt, line in enumerate(fp):
        gestures_temp, labels_temp = read_aedat31(line.split(".")[0] + ".aedat", line.split(".")[0] + "_labels.csv")
        gestures_full += gestures_temp
        labels_full += labels_temp

with open('train_dvs_gesture.pickle', 'wb') as handle:
    pickle.dump((gestures_full, labels_full), handle)




gestures_full = []
labels_full = []
with open('trials_to_test.txt') as fp:
    for cnt, line in enumerate(fp):
        gestures_temp, labels_temp = read_aedat31(line.split(".")[0] + ".aedat", line.split(".")[0] + "_labels.csv", test_set = True)
        gestures_full += gestures_temp
        labels_full += labels_temp

with open('test_dvs_gesture.pickle', 'wb') as handle:
    pickle.dump((gestures_full, labels_full), handle)




# # visualize
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# plt.clf()
# fig1 = plt.figure()

# ims = []
# for i in np.arange(sparse_matrix.shape[0]):
#     ims.append((plt.imshow( sparse_matrix[i,:,:]), ))

# im_ani = animation.ArtistAnimation(fig1, ims, interval=1, repeat_delay=2000,
#                                    blit=True)
# plt.show()

# #im_ani.save('gesture.mp4', metadata={'artist':'Clemens'})

