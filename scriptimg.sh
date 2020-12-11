for (( i = $1; i < ($1 + $2); i++ )); do
  n=$(($i*3073+1))
  cat cifar10_data/cifar-10-batches-bin/test_batch.bin | dd count=3072 bs=1 skip=$n > toto.raw
  display -size 32x32 -depth 8 -interlace Plane rgb:toto.raw &
  convert -size 32x32 -depth 8 -interlace Plane rgb:toto.raw -compress none img/cifar10_$i.ppm
  python3 src/copytest.py "img/cifar10_$i.ppm"
done
