import re
import csv

# Unpaywall citing, cited list based on mag ids 
unpaywall_citing_cited_file = open('AdditionalOutputs/unpaywallmag_references.tsv', 'w')
fieldnames = ['citing_mag_id', 'cited_mag_id']
writer = csv.DictWriter(unpaywall_citing_cited_file, delimiter="\t", fieldnames=fieldnames)

citation_pattern = re.compile(r'(=-=)([0-9]+)(-=-)')

with open('inputfiles/training_no20182019_with_contexts.txt', 'r') as file:
    for line in file:
        citing_paperid = line.split()[0]
        for citation_marker in citation_pattern.finditer(line):
            fetched_mag_id = citation_marker.group(2)
            writer.writerow({'citing_mag_id': citing_paperid,'cited_mag_id': fetched_mag_id})

