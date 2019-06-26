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

