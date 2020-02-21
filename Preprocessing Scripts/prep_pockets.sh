# Uses PyMol to define 20 angstrom cubic blocks centered at the 
# geometric origin of docked ligand pairs.

#!/bin/bash

# directory containing ligand_out.pdbqt from AutoDock Vina
cd docked
for f in *; do
	echo "Made new directory ../blocked/$f"
	mkdir -p ../blocked/$f;
	echo "Copy ligand output into ../blocked/$f"
	cp $f ../blocked/$f/ligand_out.pdbqt
	PROTEIN=`echo $f | cut -d'_' -f1`;
	echo "Copy $PROTEIN into ../blocked/$f"
	cp ../protein_pdb/$PROTEIN.pdb ../blocked/$f/protein.pdbqt
done

# run PyMol script to get block files
# conda activate py3
# directory containing ligand and pocket files
cd ../blocked
for dir in *; do
	echo "cd $dir";
	cd $dir;
	echo "pymol -c ../../get_blocks.pml";
	pymol -c ../../get_blocks.pml;
	echo "..";
	cd ..;
done
