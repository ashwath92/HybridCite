import pickle
magid_list = []
with open('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/input/mag_training_data.txt', 'r') as infile:
    for line in infile:
        magid = line.split()[0]
        magid_list.append(magid)

print(len(magid_list))
magid_set = set(magid_list)
print(len(magid_set))

with open('trainingmagids.pickle', 'wb') as pick1:
    pickle.dump(magid_set, pick1) 