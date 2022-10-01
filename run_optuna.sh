trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if test $# -ne 1; then
  echo 'run_optuna.sh column'
  exit 1
fi

num_gpus=$(nvidia-smi --list-gpus | wc -l)

column=$1
project=FB3-$column

if test -e ${project}.sqlite; then
  echo reusing existing ${project}.sqlite
else
  echo $project not found, creating new DB
  optuna create-study --study-name $project --direction minimize    --storage  sqlite:///${project}.sqlite
fi

last_gpu=$(($num_gpus - 1))

for i in $(seq 0 $last_gpu); do
  python deberta-train.py optuna cuda:$i $column &
done
wait