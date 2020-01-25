import pickle

alltrainingpapers = set()
with open('/home/ashwath/Programs/UnpaywallMAG/inputfiles/training_no20182019_with_contexts.txt', 'r') as file:
    for line in file:
        alltrainingpapers.add(line.split()[0])
with open('Pickles/unpaywallmag_training_papers.pickle', 'wb') as picc:
    pickle.dump(alltrainingpapers, picc, pickle.HIGHEST_PROTOCOL)