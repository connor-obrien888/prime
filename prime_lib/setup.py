from setuptools import setup

setup(name='primesw',
      version='0.0.1',
      description='Probabilistic Regressor for Input to the Magnetosphere Estimation',
      long_description='PRIME (Probabilistic Regressor for Input to the Magnetosphere Estimation) is a probabilistic algorithm that uses solar wind time history from L1 monitors to generate predictions of near-Earth solar wind with uncertainties.',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Astronomy',
            'Topic :: Scientific/Engineering :: Atmospheric Science'
      ],
      url='https://github.com/connor-obrien888/prime',
      author='Connor O\'Brien',
      author_email='obrienco@bu.edu',
      license='GPL-3.0',
      packages=['primesw'],
      install_requires=[
            'numpy',
            'tensorflow',
            'scikit-learn',
            'keras'],
      include_package_data=True,
      zip_safe=False)