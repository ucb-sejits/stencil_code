from distutils.core import setup

setup(
    name='stencil_code',
    version='0.95a',

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

