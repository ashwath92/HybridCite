import gensim
from tqdm import tqdm

# Write the document vectors to file
def write(infilename):
    model = gensim.models.Doc2Vec.load('MAGCompScienced2vOwnContexts.dat')
    out_f = open('MAGCompScienced2vOwnContexts.txt', 'w')
    infile = file = open(infilename, 'r')
    # The first row is the number of lines (i.e. len(model.docvecs.doctags), and the size of the vector)
    numdocvecs = len(model.docvecs)
    vector_size = model.vector_size
    out_f.write('{} {}\n'.format(numdocvecs, vector_size))
    for line in tqdm(infile):
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

def main():
    filename = '/home/ashwath/Programs/MAG-hyperdoc2vec/input/mag_training_data.txt'
    write(filename)

if __name__ == '__main__':
    main()