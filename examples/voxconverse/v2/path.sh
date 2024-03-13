. "/cm/shared/apps/anaconda3/etc/profile.d/conda.sh"

conda activate wespeaker
export PATH=$PWD:$PATH

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH
