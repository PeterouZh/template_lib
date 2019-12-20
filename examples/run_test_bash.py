import os
import sys
import argparse
import shutil

root_obs_dict = {
  'beijing': 's3://bucket-cv-competition-bj4/ZhouPeng/',
  'huanan': 's3://bucket-1892/ZhouPeng/',

}

parser = argparse.ArgumentParser()
parser.add_argument('-ro', '--root-obs', type=str, default=None, choices=list(root_obs_dict.keys()))
parser.add_argument('--port', type=int, default=6001)

if __name__ == '__main__':
  args = parser.parse_args()
  args.root_obs = root_obs_dict[args.root_obs]

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
    os.environ['DLS_TRAIN_URL'] = '/tmp'

  os.system('rm /root/.keras')
  os.system('mkdir /cache/.keras')
  os.system('ln -s /cache/.keras /root')

  os.system(command)
  pass
