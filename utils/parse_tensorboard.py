import numpy as np
import os
import glob

import tensorflow as tf


class SummaryReader(object):
  def __init__(self, tbdir):
    self.tbdir = tbdir
    self.event_paths = sorted(glob.glob(os.path.join(tbdir, "**/event*"),
                                        recursive=True))

  def get_tags(self, stop_step=10000):
    tags = set()
    for event_path in self.event_paths:
      for e in tf.train.summary_iterator(event_path):
        for v in e.summary.value:
          tags.add(v.tag)
        if e.step > stop_step:
          break
    tags = sorted(tags)
    return tags

  def get_scalar(self, tag, use_dump=True):
    data_name = tag.replace('/', '_')
    data_path = os.path.join(self.tbdir, data_name + '.pkl')
    if os.path.exists(data_path) and use_dump:
      data = np.load(data_path)
    else:
      data = []
      for event_path in self.event_paths:
        for e in tf.train.summary_iterator(event_path):
          for v in e.summary.value:
            if v.tag == tag:
              step = e.step
              data.append([step, v.simple_value])
      data = np.array(data).T
      data.dump(data_path)
    return data


def test(args, myargs):
  import matplotlib.pyplot as plt
  import pprint
  config = myargs.config.WGANGP_vs_WBGANGP_DCGAN_CelebA64
  data_dict = {}

  for label, tbdir in config.tbdirs.items():
    summary_reader = SummaryReader(tbdir=tbdir)
    tags = summary_reader.get_tags()
    myargs.logger.info(pprint.pformat(tags))
    data = summary_reader.get_scalar(tag=config.tag, use_dump=True)
    data_dict[label] = data

  plt.style.use('ggplot')
  fig, ax = plt.subplots()
  for label, data in data_dict.items():
    ax.plot(data[0], data[1], label=label)
  ax.legend()
  ax.set_ylim(config.ylim)
  ax.set_xlim(config.xlim)

  fig_name = config.tag.replace('/', '_') + '.png'
  fig.savefig(os.path.join(args.outdir, fig_name), dpi=1000)

  fig_name = config.tag.replace('/', '_') + '.pdf'
  fig.savefig(os.path.join(args.outdir, fig_name))

  pass


