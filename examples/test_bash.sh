
bs=$1
gpu=$2
export PYTHONPATH=../..
python -c "import test_bash; \
  test_bash.TestingUnit().test_resnet($bs, '$gpu')"