import os


def get_subdir2name_dict(map_file=None):
  if map_file is None:
    map_file = "template_lib/proj/imagenet/map_subdir2name.txt"
  with open(map_file, 'r') as f:
    map_list = f.readlines()
  subdir2name_dict = {}
  for line in map_list:
    line = line.strip()
    subdir, _, name = line.split(' ')
    subdir2name_dict[subdir] = name
  return subdir2name_dict

subdir2name_dict = get_subdir2name_dict()


def get_imagenet_label():
  cur_dir = os.path.dirname(__file__)
  label_file = os.path.join(cur_dir, 'imagenet_label.txt')
  id_to_label = {}
  with open(label_file) as f:
    for label_str in f.readlines():
      class_idx, name = label_str.strip('{ ,\n').split(':')
      name = name.strip("' ")
      id_to_label[int(class_idx)] = name
  return id_to_label


