import os
from subprocess import call
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from distutils.command.build import build

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

with open('README.md', 'r') as f:
  long_description = f.read()

# Instalation of C++ binaries in build mode
class CBuild(build):
  def run(self):
    # run original build code
    build.run(self)

    # install the C++ binaries
    cmd = 'make --directory={} train'.format(os.path.join(DIR_NAME, 's3d'))
    # cmd = 'make train'
    def compile():
        call(cmd.split())#, cwd='s3d')#os.path.join(DIR_NAME, 's3d'))

    self.execute(compile, [], 'Compiling C++ "train" dependecy')

# Instalation of C++ binaries in install mode
class CInstall(install):
  def run(self):
    # run original build code
    install.run(self)

    # install the C++ binaries
    cmd = 'make --directory={} train'.format(os.path.join(DIR_NAME, 's3d'))
    # cmd = 'make train'
    def compile():
        call(cmd.split())#, cwd='s3d')#os.path.join(DIR_NAME, 's3d'))

    self.execute(compile, [], 'Compiling C++ "train" dependecy')

# Instalation of C++ binaries in develop mode
class CDevelop(develop):
  def run(self):
    # run original build code
    develop.run(self)

    # install the C++ binaries
    cmd = 'make --directory={} train'.format(os.path.join(DIR_NAME, 's3d'))
    # cmd = 'make train'
    def compile():
        call(cmd.split())#, cwd='s3d')#os.path.join(DIR_NAME, 's3d'))

    self.execute(compile, [], 'Compiling C++ "train" dependecy')

# Instalation of C++ binaries in egg_info mode
class CEggInfo(egg_info):
  def run(self):
    # run original build code
    egg_info.run(self)

    # install the C++ binaries
    cmd = 'make --directory={} train'.format(os.path.join(DIR_NAME, 's3d'))
    # cmd = 'make train'
    def compile():
        call(cmd.split())#, cwd='s3d')#os.path.join(DIR_NAME, 's3d'))

    self.execute(compile, [], 'Compiling C++ "train" dependecy')


setup(
  name='pys3d',
  version='0.1.0',
  author='Peter Fennell',
  url='https://github.com/peterfennell/S3D',
  description='',
  long_description=long_description,
  packages=['s3d'],
  install_requires=[
    'pandas',
    'scikit-learn',
    'matplotlib',
    'networkx',
    'seaborn',
    'palettable',
    'joblib',
  ],
  include_package_data=True,
  cmdclass={
    'build': CBuild,
    'install': CInstall,
    'develop': CDevelop,
    'egg_info': CEggInfo,
  },
)
