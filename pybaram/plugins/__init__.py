# -*- coding: utf-8 -*-
from pybaram.plugins.base import BasePlugin
from pybaram.plugins.writer import WriterPlugin
from pybaram.utils.misc import subclass_by_name


def get_plugin(name, *args, **kwargs):
    return subclass_by_name(BasePlugin, name)(*args, **kwargs)