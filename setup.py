from setuptools import setup

setup(name='bhtsa',
      version='0.1',
      description='A Twitter Sentiment Analyzer',
      url='https://github.com/qingshuimonk/bhtsa',
      author='qingshuimonk',
      author_email='bh163@duke.edu',
      license='MIT',
      packages=['bhtsa'],
      dependency_links=['https://github.com/nltk/nltk'],
      zip_safe=False)