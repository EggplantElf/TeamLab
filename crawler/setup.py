from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='google_news_crawler',
      version='0.3.5',
      description='Google News Crawler',
      long_description=readme(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Topic :: Internet :: WWW/HTTP',
          'Topic :: Text Processing :: Linguistic',
          'Topic :: Utilities'
      ],
      keywords='Google News crawling RSS Atom download corpus creation',
      url='https://bitbucket.org/ilps/google_news_crawler',
      author='Isaac S',
      author_email='isaacsijaranamual@gmail.com',
      license='Apache License, Version 2.0',
      packages=['google_news_crawler',
                'google_news_crawler.datastore',
                'google_news_crawler.tests'],
      entry_points = {
          'console_scripts': ['google_news_crawler=google_news_crawler.google_news_crawler:main'],
      },
      install_requires=[
          'docopt',
          'elasticsearch',
          'feedparser',
          'lxml',
          'pytz',
          'pyyaml',
          'requests'
      ],
      # for stuff not registered at PyPi:
      # dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
  )
