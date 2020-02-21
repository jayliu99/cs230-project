# Gets relative side chain distances from amino acids to use in protein translation.
# Reads from the aminoacid_pdbs directory and outputs to the aminoacid_dists directory.

import os

pdbfolder = 'aminoacid_pdbs/'
distfolder = 'aminoacid_dists/'

for file in os.listdir(pdbfolder):
	with open(pdbfolder + file, 'r') as f1:
		new_f_name = distfolder + file.replace('.pdb', '.txt')
		os.makedirs(os.path.dirname(new_f_name), exist_ok=True)
		with open(new_f_name, 'w') as f2:
			diffs = [None, None, None]	#will hold x, y, z diffs
			for i, line in enumerate(f1):
				aminoacid_pdb_line = line.split()
				if i == 1:	#zero off of CA
					for j, num in enumerate(range(5, 8)):
						diffs[j] = -float(aminoacid_pdb_line[num])
				if i == 1 or i > 3: 
					if aminoacid_pdb_line[2] == 'H': 
						break
					for k, num in enumerate(range(5, 8)):
						aminoacid_pdb_line[num] = str(round(float(aminoacid_pdb_line[num]) + diffs[k], 3))
					f2.write("\t".join(aminoacid_pdb_line) + '\n')