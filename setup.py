import setuptools
setuptools.setup(name='python-vtag',
                 version='0.9.2',
                 description='Semi-Supervised Top-View Animal Tracking',
                 url='https://github.com/vt-ads/vtag',
                 python_requires='>=3.8',
                 classifiers=[
                        "Programming Language :: Python :: 3.8",
                        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                        "Operating System :: MacOS :: MacOS X",
                        "Operating System :: Microsoft :: Windows :: Windows 10"
                 ],
                 author='James Chen',
                 author_email='niche@vt.edu',
                 license='GPLv3',
                 packages=['vtag', 'vtag.core', 'vtag.gui'],
                 include_package_data=True,
                 install_requires=['numpy', 'pandas>=0.19.2',
                                   # math, models
                                   'sklearn', 'scipy', 'matplotlib',
                                   # CV
                                   'opencv-python', "opencv-contrib-python"
                                   # GUI
                                   'PyQt6',
                                   'qdarkstyle'
                                   ])
