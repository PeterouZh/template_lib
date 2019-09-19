import numpy as np
import math
import re, os
import multiprocessing


class MatPlot(object):
  def __init__(self, style='ggplot'):
    """
      plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue'])
    :param style: [classic, ggplot]
    """
    import matplotlib.pyplot as plt
    # R style
    plt.style.use(style)
    pass

  def get_fig_and_ax(self, nrows=1, ncols=1):
    """
    ax.legend(loc='best')
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    return fig, axes

  def save_to_png(self, fig, filepath, dpi=1000, bbox_inches='tight',
                  pad_inches=0.1):
    assert filepath.endswith('.png')
    fig.savefig(
      filepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

  def save_to_pdf(self, fig, filepath):
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0)

  def parse_logfile_using_re(self, logfile, re_str):
    """
    import re

    """
    with open(logfile) as f:
      logstr = f.read()
      val = [float(x) for x in re_str.findall(logstr)]
      idx = range(len(val))
    return (idx, val)


def parse_logfile(args, myargs):
  config = getattr(myargs.config, args.command)
  matplot = MatPlot()
  fig, ax = matplot.get_fig_and_ax()
  if len(config.logfiles) == 1:
    logfiles = config.logfiles * len(config.re_strs)
  for logfile, re_str in zip(logfiles, config.re_strs):
    RE_STR = re.compile(re_str)
    (idx, val) = matplot.parse_logfile_using_re(logfile=logfile, re_str=RE_STR)
    ax.plot(idx, val, label=re_str)
  ax.legend()
  matplot.save_to_png(
    fig, filepath=os.path.join(args.outdir, config.title + '.png'))
  pass


def _plot_figure(names, datas, outdir, in_one_axes=False):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  assert len(datas) == len(names)
  filename = os.path.join(outdir, 'plot_' + '__'.join(names) + '.png')
  matplot = MatPlot()
  if not in_one_axes:
    ncols = math.ceil(math.sqrt(len(names)))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = matplot.get_fig_and_ax(nrows=nrows, ncols=ncols)
    if ncols == 1 and nrows == 1:
      axes = [axes]
    else:
      axes = axes.ravel()
  else:
    ncols = 1
    nrows = 1
    fig, axes = matplot.get_fig_and_ax(nrows=nrows, ncols=ncols)
    axes = [axes] * len(names)

  for idx, (label, data) in enumerate(zip(names, datas)):
    data = data.reshape(-1, 2)
    axes[idx].plot(data[:, 0], data[:, 1], marker='.', label=label, alpha=0.7)
    axes[idx].legend(loc='best')

  matplot.save_to_png(fig=fig, filepath=filename, dpi=None, bbox_inches=None)
  plt.close(fig)
  pass


class PlotFigureProcessing(multiprocessing.Process):
  """
    worker = PlotFigureProcessing(args=(s, d, copytree))
    worker.start()
    worker.join()
  """
  def run(self):
    names, filepaths, outdir, in_one_axes = self._args
    datas = []
    for filepath in filepaths:
      data = np.loadtxt(filepath, delimiter=':')
      datas.append(data)
    _plot_figure(
      names=names, datas=datas, outdir=outdir, in_one_axes=in_one_axes)

    pass

def plot_figure(names, filepaths, outdir, in_one_axes, join=False):
  worker = PlotFigureProcessing(args=(names, filepaths, outdir, in_one_axes))
  worker.start()

  if join:
    worker.join()
  pass