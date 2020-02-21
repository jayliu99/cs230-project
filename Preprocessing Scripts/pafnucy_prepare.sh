# Prepares HDF files for running through pafnucy

DIR_OUT="datasets"

should_shuffle=0
train_frac=1
val_frac=0
prefix="data"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -s|--shuffle)
            should_shuffle=1
            shift
            ;;
        -tr|--train)
            train_frac="$2"
            shift
            shift
            ;;
        -v|--val)
            val_frac="$2"
            shift
            shift
            ;;
        -te|--test)
            test_frac="$2"
            shift
            shift
            ;;
        -p|--prefix)
            prefix="$2"
            shift
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

mkdir -p ${DIR_OUT}

olddir=$(pwd)
cd $PAFNUCY_ROOT
conda env create -f environment_gpu.yml
source activate pafnucy_env

TRAINFILE="${olddir}/${DIR_OUT}/${prefix}_train.hdf"
TESTFILE="${olddir}/${DIR_OUT}/${prefix}_test.hdf"
VALFILE="${olddir}/${DIR_OUT}/${prefix}_val.hdf"

# Stolen from https://stackoverflow.com/a/5533586
shuffle() {
   local i tmp size max rand

   # $RANDOM % (i+1) is biased because of the limited range of $RANDOM
   # Compensate by using a range which is a multiple of the array size.
   size=${#array[*]}
   max=$(( 32768 / size * size ))

   for ((i=size-1; i>0; i--)); do
      while (( (rand=$RANDOM) >= max )); do :; done
      rand=$(( rand % (i+1) ))
      tmp=${array[i]} array[i]=${array[rand]} array[rand]=$tmp
   done
}

array=($olddir/blocked/*)

if [ $should_shuffle ]
then
    shuffle
fi

size=${#array[*]}
echo "Found ${size} pairs"

let train_len=$(echo "scale=0; ${size}*${train_frac}/1" | bc)
let val_len=$(echo "scale=0; ${size}*${val_frac}/1" | bc)
let test_len=${size}-${train_len}-${val_len}

train_pockets=()
train_ligands=()
test_pockets=()
test_ligands=()
val_pockets=()
val_ligands=()

let cutoff=train_len+val_len

for ((i=size-1; i>=0; i--)); do
    id="$(basename -- ${array[i]})"
    id="${id%.*}"
    cp "${array[i]}/pocket.mol2" "${array[i]}/pocket_$id.mol2"
    cp "${array[i]}/ligand.mol2" "${array[i]}/ligand_$id.mol2"
    if [ $i -lt $train_len ]; then
        train_pockets+=("${array[i]}/pocket_$id.mol2")
        train_ligands+=("${array[i]}/ligand_$id.mol2")
    elif [ $i -lt $cutoff ]; then
        val_pockets+=("${array[i]}/pocket_$id.mol2")
        val_ligands+=("${array[i]}/ligand_$id.mol2")
    else
        test_pockets+=("${array[i]}/pocket_$id.mol2")
        test_ligands+=("${array[i]}/ligand_$id.mol2")
    fi
done

echo "Preparing training data (${#train_pockets[*]} pairs)"
python3 prepare.py -l ${train_ligands[@]} -p ${train_pockets[@]} -o \
    "$TRAINFILE" --affinities "$olddir/data.txt"

if [ $val_len -gt 0 ]
then
    echo ""
    echo "Preparing validation data (${#val_pockets[*]} pairs)"
    python3 prepare.py -l ${val_ligands[@]} -p ${val_pockets[@]} -o "$VALFILE" \
        --affinities "$olddir/affinities.csv"
fi

if [ $test_len -gt 0 ]
then
    echo ""
    echo "Preparing test data (${#test_pockets[*]} pairs)"
    python3 prepare.py -l ${test_ligands[@]} -p ${test_pockets[@]} -o \
        "$TESTFILE" --affinities "$olddir/affinities.csv"
fi

echo "Cleaning up temp files"
for ((i=size-1; i>0; i--)); do
    id="$(basename -- ${array[i]})"
    id="${id%.*}"
    rm "${array[i]}/pocket_$id.mol2"
    rm "${array[i]}/ligand_$id.mol2"
done
