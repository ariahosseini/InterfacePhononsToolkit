from setuptools import setup, find_packages
setup(name='InterfacePhononsToolkit',
      version='0.1',
      description='Interface Phonons Toolkit',
      url='#',
      author='S.Aria Hosseini',
      author_email='shoss008@ucr.edu',
      zip_safe=False,
      package_dir={"": "util"},
      packages=find_packages(where="util"),
      python_requires=">=3.6",
      )
