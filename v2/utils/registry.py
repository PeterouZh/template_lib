from typing import Dict, Optional
import logging
from fvcore.common.registry import Registry as Registry_base


class Registry(Registry_base):

  def _do_register(self, name: str, obj: object) -> None:
    assert (
          name not in self._obj_map
    ), "An object named '{}' was already registered in '{}' registry!".format(
      name, self._name
    )
    # logging.getLogger('tl').warning(f"\n  {name} was already registered in {self._name} registry!")
    self._obj_map[name] = obj
    pass

  def register(self, obj: object = None, name=None, name_prefix=None) -> Optional[object]:
    """
    Register the given object under the the name `obj.__name__`.
    Can be used as either a decorator or not. See docstring of this class for usage.
    """
    if obj is None:
      # used as a decorator
      def deco(func_or_class: object, name=name, name_prefix=name_prefix) -> object:
        # name = func_or_class.__name__  # pyre-ignore
        if name is None:
          if name_prefix is None:
            name = f"{func_or_class.__module__}.{func_or_class.__name__}"
          else:
            name = f"{name_prefix}.{func_or_class.__name__}"
        else:
          name = name
        self._do_register(name, func_or_class)
        return func_or_class

      return deco