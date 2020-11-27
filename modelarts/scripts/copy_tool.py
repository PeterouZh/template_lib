import os
import shutil
import argparse

import moxing as mox

parser = argparse.ArgumentParser(description='copy tool')
parser.add_argument('-s', type=str)
parser.add_argument('-d', type=str)
parser.add_argument('-t', type=str, default="copy", choices=['copytree', 'copy',
                                                             'copytree_nooverwrite', 'copy_nooverwrite'])
cfg = parser.parse_args()

def main():
    print('=> Copy file(s) from %s to %s ...' % (cfg.s, cfg.d))
    if cfg.t == "copytree":
        mox.file.copy_parallel(cfg.s, cfg.d)
    elif cfg.t == 'copy':
        mox.file.copy(cfg.s, cfg.d)
    elif cfg.t == 'copytree_nooverwrite':
        if os.path.exists(cfg.d):
            print('Skip copying, %s exist!' % cfg.d)
            return
        mox.file.copy_parallel(cfg.s, cfg.d)
    elif cfg.t == 'copy_nooverwrite':
        if os.path.exists(cfg.d):
            print('Skip copying, %s exist!' % cfg.d)
            return
        mox.file.copy_parallel(cfg.s, cfg.d)
    else:
        assert 0
    print('=> End copy file(s) from %s to %s ...' % (cfg.s, cfg.d))

if __name__ == '__main__':
    main()
