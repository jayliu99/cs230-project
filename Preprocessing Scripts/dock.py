# Docks the ligands to the proteins.
# Takes files from the protein_pdbqt and ligand_pdbqt directories and outputs files 
# in the docked directory. It uses the configuration file conf.txt. Logs are placed 
# in the vina_logs directory.

import csv
import os
import subprocess
import sys

PROTEIN_DIR = 'protein_pdbqt'
LIGAND_DIR = 'ligand_pdbqt'
OUTDIR = 'docked'
LOGDIR = 'vina_logs'
DATA_FILE = 'data.txt'
CONFIG = 'conf.txt'


def dock(overwrite):
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    vina = "{}/bin/vina".format(os.environ['VINA_ROOT'])

    errs = []

    with open(DATA_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        rows = list(reader)
        progress = 0
        for row in rows[1:]:
            ligand = row[1]
            protein = row[3]
            progress += 1
            protein_id = protein.strip().replace('/', '.').replace(' ', '.')
            protein_file = "{}/{}.pdbqt".format(PROTEIN_DIR, protein_id)
            ligand_file = "{}/{}.pdbqt".format(LIGAND_DIR, ligand)
            logfile = "{}/{}_{}.txt".format(LOGDIR, protein_id, ligand)
            outfile = "{}/{}_{}.pdb".format(OUTDIR, protein_id, ligand)
            if not overwrite and os.path.exists(outfile):
                print("Already docked pair {}/{}: {} to {}".format(progress,
                        len(rows) - 1, ligand, protein))
            else:
                print("Docking pair {}/{}: {} to {}".format(progress,
                        len(rows) - 1, ligand, protein))
                try:
                    out = subprocess.check_output([vina, '--receptor',
                            protein_file, '--ligand', ligand_file, '--config',
                            CONFIG, '--out', outfile], stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    out = e.output
                    errs.append((e, protein, ligand))
                finally:
                    with open(logfile, 'a', encoding='utf-8') as log:
                        log.write(out.decode('utf-8'))

    if len(errs) > 0:
        err = '\n' + '-' * 72 + '\n'
        err += 'Errors occurred while docking:\n'
        for e, protein, ligand in errs:
            err += "\nError docking ligand {} to receptor {}: {}\n".format(
                    ligand, protein, e)
        sys.exit(err)


def main():
    overwrite = ('-o' in sys.argv)
    dock(overwrite)


if __name__ == '__main__':
    main()
