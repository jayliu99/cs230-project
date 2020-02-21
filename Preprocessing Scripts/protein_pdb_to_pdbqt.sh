# Converts protein PDB files into PDBQT files

DIR_IN="protein_pdb"
DIR_OUT="protein_pdbqt"

mkdir -p $DIR_OUT

for file in $DIR_IN/*; do
  echo "Converting file $file"
  outfile="$(basename -- $file)"
  outfile="$DIR_OUT/${outfile%.*}.pdbqt"
  $MGL_ROOT/bin/python \
    "$MGL_ROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py" \
    -r "$file" -o "$outfile" -A hydrogens
done
