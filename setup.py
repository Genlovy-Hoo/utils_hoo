# -*- coding: utf-8 -*-

from distutils.core import setup

with open('./readme.txt') as f:
    readme = f.read()

setup(name='utils_hoo',
      version='0.1.3',
      author='Genlovy Hoo',
      author_email='genlovhyy@163.com',
      url='www.genlovy.cn',
      license='MPL 2.0',
      description="Genlovy Hoo's utils.",
      long_description=readme,
      platform='any',
      packages=['utils_hoo',
                'utils_hoo.utils_logging',
                'utils_hoo.utils_plot',
                'utils_hoo.utils_datsci',
                'utils_hoo.utils_fin',
				'utils_hoo.utils_optimizer'])
