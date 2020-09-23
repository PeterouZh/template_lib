import matplotlib
matplotlib.use('agg')
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
            data_xlim = None
            axes_prop = default_dict.get('properties')
            if axes_prop is not None:
              if 'xlim' in axes_prop:
                data_xlim = axes_prop['xlim'][-1]

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
                    # limit x in a range
                    if data_xlim:
                      data = data[data[:, 0] <= data_xlim]
                    
                    itr_val_str = self.get_itr_val_str(data, show_max[idx])
                    label_str = f'{itr_val_str}' + f'-{modi_minutes:03d}m---' + label
                    
                    axes[idx].plot(data[:, 0], data[:, 1], label=label_str, marker='.', linewidth='5', markersize='10', alpha=0.5)
                    label2datas[label] = data
            axes[idx].legend(prop={'size': legend_size})
            axes[idx].set(**default_dict['properties'])
                    
            label2datas_list.append(label2datas)
        fig.savefig(outfigure, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        return label2datas_list


class TestingPlot(unittest.TestCase):

  def test_plot_FID_CBN_location(self):
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

    from datetime import datetime
    TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
    time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
    outdir = outdir if not TIME_STR else (outdir + '_' + time_str)

    from template_lib.utils.plot_results import PlotResults
    import collections, shutil

    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    outfigure = os.path.join(outdir, 'top1.jpg')
    default_dicts = []
    show_max = []

    top1 = collections.defaultdict(dict)
    top1['results/CIFAR10/train_cifar10_20200410-13_41_09_498'] = \
      {'CBN_all_blocks': 'textlog/evaltf.ma0.FID_tf.log', }
    top1['properties'] = {'title': 'top1', }
    default_dicts.append(top1)
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    pass

  def test_plot_FID_cifar10_style_position(self):
    """
    python -c "from exp.tests.test_styleganv2 import Testing_stylegan2_style_position;\
      Testing_stylegan2_style_position().test_plot_FID_cifar10_style_position()"
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file none
                    --tl_command none
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from template_lib.utils.plot_results import PlotResults
    import collections

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = []
    show_max = []

    FID = collections.defaultdict(dict)
    FID['results/stylegan2_style_position/train_cifar10_style_position_20200906-12_02_45_391'] = \
      {'12_02_45_391-all_position': 'textdir/train.ma0.FID_tf.log', }

    FID['properties'] = {'title': 'FID', }
    default_dicts.append(FID)
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')
    pass
