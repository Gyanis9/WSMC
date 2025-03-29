# shellcheck disable=SC2155
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$1 python3 main.py --task_name FewRel

