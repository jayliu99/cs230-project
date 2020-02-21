# Gets the PDB files of translations from hNaV1.7.
# Reads from the aminoacid_dists directory and the 
# base_protein.pdb and protein_translations.csv files and outputs to the protein_pdb directory.

import os

translationsfolder = 'protein_pdb/'
distfolder = 'aminoacid_dists/'
aminoacids = {'T': 'thr', 'S': 'ser', 'A': 'ala', 'E': 'glu', 'Q': 'gln', 'V': 'val', 'I': 'ile',\
			  'C': 'cys', 'D': 'asp', 'K': 'lys', 'Y': 'tyr', 'M': 'met', 'N': 'asn', 'F': 'phe',\
			  'R': 'arg', 'L': 'leu'}


def align_with_spaces(str_width, curr_str, left_align=False): #default right align
	curr_width = len(curr_str)
	for _ in range(str_width - curr_width):
		if left_align:
			curr_str = curr_str + ' '
		else:
			curr_str = ' ' + curr_str
	return curr_str


with open('protein_translations.csv', 'r') as translations_file:
	for w, line in enumerate(translations_file):
		if w == 0:
			continue
		translation_line = line.split(',')
		ID = translation_line[0].strip().replace('/', '.').replace(' ', '.')
		new_f_name = translationsfolder + ID + '.pdb'
		os.makedirs(os.path.dirname(new_f_name), exist_ok=True)
		changes = translation_line[1].strip().split('/')
		with open(new_f_name, 'w') as new_file:
			with open('base_protein.pdb', 'r') as base:
				if changes[0] == '-':
					for base_pdb_line in base:
						new_file.write(base_pdb_line)
					continue
				changes = sorted(changes, key=lambda x: int(x[1:-1]))
				count = 0
				origin = [None, None, None]
				aa_start = 4
				number = int(changes[0][1:-1])
				for base_pdb_line in base:
					record_name = base_pdb_line[0:6]
					if record_name == 'ANISOU':
						new_file.write(record_name + str(count) + base_pdb_line[11:])
						continue 
					if record_name != 'ATOM  ': 	#TER / END
						new_file.write(base_pdb_line)
						continue
					serial = base_pdb_line[6:11]
					name = base_pdb_line[12:16]
					altLoc = base_pdb_line[16]
					resSeq = base_pdb_line[22:26]
					x = base_pdb_line[30:38]
					y = base_pdb_line[38:46]
					z = base_pdb_line[46:54]
					element = base_pdb_line[76:78]
					charge = base_pdb_line[78:80]
					if int(resSeq) > number:
						aa_start = 4
						if len(changes) > 0:
							changes.pop(0)
					if len(changes) > 0:
						number = int(changes[0][1:-1])
						if int(resSeq) < number:
							count += 1
							new_file.write(record_name + \
								align_with_spaces(len(serial), str(count)) + \
								base_pdb_line[11:])
						elif int(resSeq) == number:
							aa_start -= 1
							if aa_start >= 0:
								if name.strip() == 'CA':
									for i, var in enumerate([x, y, z]):
										origin[i] = float(var)
								count += 1
								new_file.write(record_name + \
									align_with_spaces(len(serial), str(count)) + \
									' ' + \
									name + \
									altLoc + \
									aminoacids[changes[0][-1]].upper() + \
									base_pdb_line[20:])
							elif aa_start == -1:
								aa_name = aminoacids[changes[0][-1]]
								with open(distfolder + aa_name + '.txt', 'r') as dist_file:
									for j, dist_line in enumerate(dist_file):
										if j > 0:
											dist_line_lst = dist_line.split()
											count += 1
											new_file.write(record_name + \
												align_with_spaces(len(serial), str(count)) + \
												' ' + \
												align_with_spaces(len(name), align_with_spaces(len(name) - 1, dist_line_lst[2], left_align=True)) + \
												' ' + \
												dist_line_lst[3] + \
												base_pdb_line[20:30] + \
												align_with_spaces(len(x), str(round(origin[0] + float(dist_line_lst[5]), 3))) + \
												align_with_spaces(len(y), str(round(origin[1] + float(dist_line_lst[6]), 3))) + \
												align_with_spaces(len(z), str(round(origin[2] + float(dist_line_lst[7]), 3))) + \
												base_pdb_line[54:76] + \
												align_with_spaces(len(element), dist_line_lst[10][0]) + \
												align_with_spaces(len(charge), dist_line_lst[10][1:]) + \
												'\n')
							else:
								continue
					else: 
						count += 1
						new_file.write(record_name + align_with_spaces(len(serial), str(count)) + base_pdb_line[11:])
		
