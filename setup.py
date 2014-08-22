from distutils.core import setup

setup(
    name='stencil_code',
    description='A specializer built on the ASP SEJITS framework',
    url='https://github.com/ucb-sejits/stencil_code/',
    version='0.95a9',

    packages=[
        'stencil_code',
        'stencil_code.backend',
    ],

    package_data={
        'stencil_code': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
        'numpy',
    ]
)
