import sys,os
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))

#os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(sys.path)

__all__ = ["utils","models","smpl_lib","trainer"]

from .tailornet import TailorNet