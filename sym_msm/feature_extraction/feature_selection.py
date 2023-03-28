from ENPMDA.analysis import *
from ENPMDA.analysis.base import DaskChunkMdanalysis

from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.analysis.distances import dist
from MDAnalysis.core.groups import AtomGroup, ResidueGroup

import MDAnalysis as mda
import itertools
import pandas as pd

# find the next subunit from the current subunit
# it is specific to different systems
subunit_iter_dic = {'A': 'B',
                    'B': 'C',
                    'C': 'D',
                    'D': 'E',
                    'E': 'A',
                    'F': 'G',
                    'G': 'H',
                    'H': 'I',
                    'I': 'J',
                    'J': 'F'}

class get_c_alpha_distance_10A(DaskChunkMdanalysis):
    name = 'ca_distance_10A'
    
    def set_feature_info(self, universe):
        # the creation of `pair_indices_union_df` can be found in example
        pair_indices_union_df = pd.read_pickle('pair_indices_union_df.pickle')
        feat_info = []
        ag1 = universe.atoms[[]]
        ag2 = universe.atoms[[]]
        for subunit in range(5):
            for ind, row in pair_indices_union_df.iterrows():
                ag1 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a1_chain, row.a1_resid))
                ag2 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a2_chain, row.a2_resid))
            pair_indices_union_df = pair_indices_union_df.replace({"a1_chain": subunit_iter_dic})
            pair_indices_union_df = pair_indices_union_df.replace({"a2_chain": subunit_iter_dic}) 
            
        for ca_ag1, ca_ag2 in zip(ag1, ag2):
            feat_info.append(f'{ca_ag1.segid}_{ca_ag1.resid}_{ca_ag2.segid}_{ca_ag2.resid}')
        self.ag1_indices = ag1.indices
        self.ag2_indices = ag2.indices
        return feat_info

    def run_analysis(self, universe, start, stop, step):
        result = []
        ag1 = universe.atoms[self.ag1_indices]
        ag2 = universe.atoms[self.ag2_indices]
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(ag1, ag2)[2])
        return result


class get_c_alpha_distance_10A_2diff(DaskChunkMdanalysis):
    name = 'ca_distance_10A_2diff'
    
    def set_feature_info(self, universe):
        # the creation of `pair_indices_union_df` can be found in example
        pair_indices_union_df = pd.read_pickle('pair_indices_union_df_2div.pickle')
        feat_info = []
        ag1 = universe.atoms[[]]
        ag2 = universe.atoms[[]]
        for subunit in range(5):
            for ind, row in pair_indices_union_df.iterrows():
                ag1 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a1_chain, row.a1_resid))
                ag2 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a2_chain, row.a2_resid))
            pair_indices_union_df = pair_indices_union_df.replace({"a1_chain": subunit_iter_dic})
            pair_indices_union_df = pair_indices_union_df.replace({"a2_chain": subunit_iter_dic}) 
            
        for ca_ag1, ca_ag2 in zip(ag1, ag2):
            feat_info.append(f'{ca_ag1.segid}_{ca_ag1.resid}_{ca_ag2.segid}_{ca_ag2.resid}')
        self.ag1_indices = ag1.indices
        self.ag2_indices = ag2.indices
        return feat_info

    def run_analysis(self, universe, start, stop, step):
        result = []
        ag1 = universe.atoms[self.ag1_indices]
        ag2 = universe.atoms[self.ag2_indices]
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(ag1, ag2)[2])
        return result