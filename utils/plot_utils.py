import re, os
import matplotlib.pyplot as plt


class MatPlot(object):
  def __init__(self):
    # R style
    plt.style.use('ggplot')
    pass

  def get_fig_and_ax(self):
    fig, ax = plt.subplots()
    return fig, ax

  def save_to_png(self, fig, filepath):
    assert filepath.endswith('.png')
    fig.savefig(filepath, dpi=1000, bbox_inches='tight', pad_inches=0)

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
