import gensim

infilename = '/home/ashwath/Programs/UnpaywallMAG/inputfiles/training_no20182019_with_contexts.txt'
# Write the document vectors to file
model = gensim.models.Doc2Vec.load('UnpaywallMagd2v.dat')
out_f = open('UnpaywallMagd2v.txt', 'w')
infile = file = open(infilename, 'r')
out_f.write('{} {}\n'.format(len(model.docvecs.offset2doctag), model.vector_size))
for line in infile:
    paperid = line.split()[0]
    try:
        vect = model.docvecs[paperid]
    except KeyError:
        continue
    str_vect = ' '.join([str(j) for j in vect])
    string = "{} {}".format(paperid, str_vect) 
    out_f.write(string + '\n')
out_f.close()
infile.close()


