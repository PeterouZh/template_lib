import matplotlib.pyplot as plt
import numpy as np
import os
import unittest


class PlotResults(object):
    
    def __init__(self, ):
        # PlotResults.setup_env()
        
        pass
    
    @staticmethod
    def setup_env():
        import os
        try:
            import mpld3
        except:
            os.system('pip install mpld3')
    
    def get_last_md_inter_time(self, filepath):
        from datetime import datetime, timedelta

        modi_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        modi_inter = datetime.now() - modi_time
        modi_minutes = modi_inter.total_seconds() // 60
        return int(modi_minutes)
    
    def get_fig_axes(self, rows, cols, figsize_wh=(15, 7)):
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'lime', 'tan', 'salmon', 'gold', 'darkred', 'darkblue'])
        fig, axes = plt.subplots(rows, cols, figsize=(figsize_wh[0]*cols, figsize_wh[1]*rows))
        if rows * cols > 1:
            axes = axes.ravel()
        else:
            axes = [axes]
        return fig, axes
    
    def get_itr_val_str(self, data, ismax):
        if ismax:
            itr = int(data[:, 0][data[:, 1].argmax()])
            val = data[:, 1].max()
            return f'itr.{itr:06d}_maxv.{val:.3f}'
        else:
            itr = int(data[:, 0][data[:, 1].argmin()])
            val = data[:, 1].min()
            return f'itr.{itr:06d}_minv.{val:.3f}'

    def plot_defaultdicts(self, outfigure, default_dicts, show_max=True, figsize_wh=(15, 8), legend_size=12,
                          dpi=500, ):

        import tempfile
        if not isinstance(default_dicts, list):
            default_dicts = [default_dicts]
        if not isinstance(show_max, list):
            show_max = [show_max]
        assert len(show_max) == len(default_dicts)

        fig, axes = self.get_fig_axes(rows=len(default_dicts), cols=1, figsize_wh=figsize_wh)
        
        label2datas_list = []
        for idx, default_dict in enumerate(default_dicts):
            label2datas = {}
            # for each result dir
            for (result_dir, label2file) in default_dict.items():
                if result_dir == 'properties':
                    continue
                # for each texlog file
                for label, file in label2file.items():
                    filepath = os.path.join(result_dir, file)
                    if not os.path.exists(filepath):
                      print(f'Not exist {filepath}, skip.')
                      continue
                    # get modified time
                    modi_minutes = self.get_last_md_inter_time(filepath)

                    data = np.loadtxt(filepath, delimiter=':')
                    data = data.reshape(-1, 2)
                    
                    itr_val_str = self.get_itr_val_str(data, show_max[idx])
                    label_str = f'{itr_val_str}' + f'-{modi_minutes:03d}m---' + label
                    
                    axes[idx].plot(data[:, 0], data[:, 1], label=label_str, marker='.', linewidth='5', markersize='15', alpha=0.5)
                    label2datas[label] = data
            axes[idx].legend(prop={'size': legend_size})
            axes[idx].set(**default_dict['properties'])
                    
            label2datas_list.append(label2datas)
        fig.savefig(outfigure, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        return label2datas_list


class TestingPlot(unittest.TestCase):

  def test_only_conv3(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'


    from template_lib.utils.plot_results import PlotResults
    import collections, shutil

    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    top1 = collections.defaultdict(dict)
    outfigure = os.path.join(outdir, 'top1.jpg')

    top1['results/Exp/search_with_only_sep_conv3_op_20200229-23_18_13_824'] = \
      {'ops.sepConv3_affinel.f'      : 'textlog/valid.ma1.top1.log', }
    top1['results/Exp/search_with_only_sep_conv3_op_20200229-23_19_39_828'] = \
      {'ops.sepConv3_affinel.t': 'textlog/valid.ma1.top1.log', }
    top1['results/Exp/search_with_only_conv3_op_20200229-23_20_25_557'] = \
      {'ops.conv3_affinel.f': 'textlog/valid.ma1.top1.log', }
    top1['results/PTDARTS/darts_search_20200229-23_20_54_914'] = \
      {'ops.darts8ops_affinel.f': 'textlog/valid.ma1.top1.log', }

    top1['properties'] = {'title': 'top1', }
    default_dicts = [top1]
    show_max = [True, ]
    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    pass
