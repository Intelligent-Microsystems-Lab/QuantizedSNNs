import numpy as np
import math
import pickle
import os


def dat2mat(filename, retinaSizeX, only_pos=False):
    """dat2mat.py: This script converts a aedat file into a list of events.
        It only works for 32 unsinged values in the aedat file.

    filename: name of the dat file
    retinaSizeX: one dimension of the retina size
    only_pos: True to delete all the negative spikes from the dat file
    """
    print('Addresses will be interpreted as 32 bits')
    maxEvents = 30e6
    numBytesPerEvent = 8

    f = open(filename, 'r', encoding='latin-1')
    bof = f.tell()
    #Skip header lines
    line = f.readline()
    while (line[0]=='#'):
        print(line)
        bof = f.tell()
        line = f.readline()

    #Calculate number of events
    f.seek(0,2) #EOF
    numEvents = (f.tell()-bof)/numBytesPerEvent
    if (numEvents>maxEvents):
        print("More events than the maximum events!!!")
        numEvents = maxEvents
    #Read data
    f.seek(bof,0)
    dataArray = np.fromfile(f, '>u4')
    allAddr = dataArray[::2]
    allTs = dataArray[1::2]
    f.close()
    #print allTs

    #Define event format
    xmask = 0xFE
    ymask = 0x7F00
    xshift = 1
    yshift = 8
    if (retinaSizeX == 32):
        xshift=3 #Subsampling of 4
        yshift=10 #Subsampling of 4
    polmask = 0x1
    addr = abs(allAddr)
    x = (addr & xmask)>>xshift
    y = (addr & ymask)>>yshift
    pol = 1 - (2*(addr & polmask)) #1 for ON, -1 for OFF
    pol = pol.astype(np.int32)
    '''    
    #invert x
    x = retinaSizeX - x
    '''
    #Do relative time
    tpo = allTs;
    tpo[:] = tpo[:]-tpo[0]

    stim = np.array([tpo, np.zeros(x.size, dtype=np.int), \
        -1*np.ones(x.size, dtype=np.int), x, y, pol])
    stim = np.transpose(stim)
    
    if (only_pos == True):
        res_stim = stim[stim[:,5]==1, :]
    else:
        res_stim = stim

    # bin them 1ms
    res_stim[:,0] = np.floor(res_stim[:,0]/1000)
    #res_stim[:,0] = res_stim[:,0] - np.min(res_stim[:,0])

    return res_stim
  

chunk_size = 700
#chunk_size = 1500
#chunk_size = 2500
file_list = ["RetinaTeresa2-club_long.aedat", "RetinaTeresa2-diamond_long.aedat", "RetinaTeresa2-heart_long.aedat", "RetinaTeresa2-spade_long.aedat"]
start_ts = np.arange(0,121000/chunk_size)*chunk_size
end_ts = np.arange(0,121000/chunk_size)*chunk_size + chunk_size #its not 3min... one recording is just 2min!
cards_full = []
labels_full = []

for idx,cur_file in enumerate(file_list):
    stim_cur = dat2mat(cur_file, 128, False)
    for i in np.arange(len(start_ts)):
        temp_cur = stim_cur[stim_cur[:,0] >= start_ts[i]]
        temp_cur = temp_cur[temp_cur[:,0] < end_ts[i]]
        if(len(temp_cur) == 0):
            import pdb; pdb.set_trace()
        temp_cur[:,0] = temp_cur[:,0]-start_ts[i]
        cards_full.append(temp_cur)
    labels_full += [idx]*len(start_ts)

#80/20 split train/test
cards_full = np.array(cards_full)
labels_full = np.array(labels_full)
shuffle_idx = np.arange(len(labels_full))
np.random.shuffle(shuffle_idx)
cards_full = cards_full[shuffle_idx]
labels_full = labels_full[shuffle_idx]


with open('slow_poker_'+str(chunk_size)+'_train.pickle', 'wb') as handle:
    pickle.dump((cards_full[:int(len(labels_full)*.8)  ], labels_full[:int(len(labels_full)*.8)  ]), handle)
with open('slow_poker_'+str(chunk_size)+'_test.pickle', 'wb') as handle:
    pickle.dump((cards_full[int(len(labels_full)*.8):], labels_full[int(len(labels_full)*.8):]), handle)


