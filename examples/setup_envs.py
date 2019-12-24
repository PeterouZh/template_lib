import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-ro', '--root-obs', '--root_obs', type=str, default='s3://bucket-xx')
parser.add_argument('--env', type=str, default='')
parser.add_argument('-t', '-copy_type', type=str, default='copytree', choices=['copytree', 'copytree_nooverwrite'])


def setup_package(packages):
  command_template = 'pip install %s'
  for pack in packages:
    command = command_template % pack
    print('=Installing %s'%pack)
    os.system(command)


def detectron2():
  print('=setup detectron2 envs:')
  packs = ['termcolor==1.1.0', 'portalocker==1.5.2', 'yacs>=0.1.6']
  setup_package(packs)
  os.system('tar -zxvf cocoapi.tar.gz')
  os.system('cd cocoapi/PythonAPI && python setup.py install')
  os.system('tar -zxvf fvcore.tar.gz')
  os.system('cd fvcore && python setup.py install')
  
  pass


if __name__ == '__main__':
  args, unparsed = parser.parse_known_args()

  if not args.env:
    exit(0)

  try:
      import moxing

      pypi_obs = os.path.join(args.root_obs, 'pypi', args.env)
      pypi = os.path.join('/cache/pypi', args.env)
      os.system('python copy_tool.py -s {s} -d {d} -t {t}'.format(s=pypi_obs, d=pypi, t=args.t))
      # moxing.file.copy_parallel(pypi_obs, pypi)

  except ImportError:
      pass
  except Exception as e:
      if str(e) == 'server is not set correctly':
          pass
      else:
          raise e
  finally:
      os.chdir(pypi)

  eval('{func}()'.format(func=args.env))
  print('==End setup_envs: %s'%args.env)
