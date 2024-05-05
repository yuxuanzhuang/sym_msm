====================
Symmetry-adapted MSM
====================

This package contains essential tools to perform markov state models on symmetric systems e.g. a homopentameric ligand-gated ion channel.

Installation
------------
The package can be installed by running the following command:

    git clone --recurse-submodules https://github.com/yuxuanzhuang/sym_msm.git
    cd sym_msm
    pip install .

Folder
------
- msm: `MSMInitializer` base class to load the trajectory features,
    running feature decomposition (e.g. for symTICA, `decomposition.SymTICAInitializer`),
    clustering, and build Markov state model.
- decomposition: Code to perform `SymTICA` decomposition.
    It also includes symmetry-aware multidimensional scaling (MDS) to further 
    visualize the asymmetry presented in the transformed symTICA space.
- feature_extraction: Feature extraction script of C-alpha distances
and sort them in blocks of symmetry-related residues.
- util: Utility functions.


Usage
-----
See the jupyter notebook in the example folder.

- example_symtica.ipynb: This notebook demonstrates
how to use `SymTICA` to extract the slowest modes of a symmetric system.

- example_sym_msm.ipynb: This notebook demonstrates
how to use `SymMSM` to build a symmetry-aware Markov state model.

- example_sym_mds.ipynb: This notebook demonstrates
how to use symmetry-aware multidimensional scaling
to visualize the asymmetry presented in the data.

* Free software: 3-clause BSD license