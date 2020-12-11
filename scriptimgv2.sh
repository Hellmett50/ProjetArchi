for (( i = $1; i < ($1 + $2); i++ )); do
  n0=$(($i*3073))
  n=$(($i*3073+1))
  name=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")
  cat cifar10_data/cifar-10-batches-bin/test_batch.bin | dd count=1 bs=1 skip=$n0 | xxd > temp
  pos=11
  len=1
  nameindex=$( cat temp )
  #echo "nameindex=${nameindex:${pos}}"
  cat cifar10_data/cifar-10-batches-bin/test_batch.bin | dd count=3072 bs=1 skip=$n > toto.raw
  convert -size 32x32 -depth 8 -interlace Plane rgb:toto.raw -compress none img/cifar10_${i}_${name[${nameindex:$pos:$len}]}.ppm
  #display img/cifar10_${i}.ppm &
  python3 src/copytest.py "img/cifar10_${i}_${name[${nameindex:$pos:$len}]}.ppm"
done
