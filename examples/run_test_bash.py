import os
import sys
import argparse
import shutil

root_obs_dict = {
  'beijing': 's3://bucket-cv-competition-bj4/ZhouPeng/',
  'huanan': 's3://bucket-1892/ZhouPeng/',

}

parser = argparse.ArgumentParser()
parser.add_argument('-ro', '--root-obs', '--root_obs', type=str, default=None, choices=list(root_obs_dict.keys()))
parser.add_argument('--port', type=int, default=6001)
parser.add_argument('--exp', type=str, default='')


def setup_package():
  packages = ['pyyaml==5.2', 'easydict', 'tensorboardX==1.9']
  command_template = 'pip install %s'
  for pack in packages:
    command = command_template % pack
    print('=Installing %s'%pack)
    os.system(command)


if __name__ == '__main__':
  # args = parser.parse_args()
  args, unparsed = parser.parse_known_args()
  args.root_obs = root_obs_dict[args.root_obs]

  setup_package()

  try:
    import moxing
    code_obs = os.path.join(args.root_obs, 'code')
    code = '/cache/code'
    print('Copying code from [%s] to [%s]'%(code_obs, code))
    moxing.file.copy_parallel(code_obs, code)
  except ImportError:
    pass
  except Exception as e:
    if str(e) == 'server is not set correctly':
      pass
    else:
      raise e
  finally:
    os.chdir(os.path.join(code, 'template_lib/examples'))

  cwd = os.getcwd()
  print('cwd: %s'%cwd)

  command = '''
        export CUDA_VISIBLE_DEVICES=0
        export PORT={port}
        export TIME_STR=1
        export PYTHONPATH=../..
        python -c "import test_bash; \
          test_bash.TestingUnit().test_bash($PORT)"
        '''.format(port=args.port)


  os.environ['RESULTS_OBS'] = os.path.join(args.root_obs, 'results/template_lib')
  if 'DLS_TRAIN_URL' not in os.environ:
    os.environ['DLS_TRAIN_URL'] = '/tmp/logs/1'

  os.system('rm /root/.keras')
  os.system('mkdir /cache/.keras')
  os.system('ln -s /cache/.keras /root')

  try:
    os.system(command)
  except:
    pass
  pass
