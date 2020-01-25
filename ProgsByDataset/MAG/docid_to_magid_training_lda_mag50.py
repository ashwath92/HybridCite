import pickle

docid_to_magid = dict()
with open('/home/ashwath/Programs/MAGCS/MAG-hyperdoc2vec/input/mag_training_data_50citationsmin.txt', 'r') as file:
    for i, line in enumerate(file):
        docid_to_magid[i] = line.split()[0]

with open('docid_to_magid_training_mag50.pickle', 'wb') as pick:
    pickle.dump(docid_to_magid, pick)
    print("Pickled")