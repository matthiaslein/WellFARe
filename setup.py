from setuptools import setup

setup(
    name='wellfare',
    version='1.0.3',
    python_requires='>=3.4',
    install_requires=['scipy', 'numpy', 'matplotlib', 'typing'],
    packages=['wellfare'],
    scripts={'wellfare/comparestruct', 'wellfare/freeda', 'wellfare/mincepies',
             'wellfare/specplot', 'wellfare/wellfare'},
    url='https://github.com/matthiaslein/WellFARe',
    license='GNU General Public License v3.0',
    author='Matthias Lein',
    author_email='matthias.lein@gmail.com',
    description='WellFARe - The Wellington Fast Assessment of REactions',
    long_description='The wellfare package is a collection of quantum chemical'
                     ' programs that interact with electronic structure'
                     ' packages in order to aid analysis and computation. The'
                     ' distribution includes a tool to compare 3d structures,'
                     ' the Free Energy Decomposition Analysis, the mincepies'
                     ' tool for conformer generation and the wellfare force'
                     ' field itself.'
)
