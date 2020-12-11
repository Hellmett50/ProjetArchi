for (( i = $1; i < ($1 + $2); i++ )); do
  python3 src/copytest.py "img/cifar10_${i}*"
done
