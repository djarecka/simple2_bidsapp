#!/usr/bin/env python
"""Run a demo workflow that retrieves brain images and processes them

"""
import os

from nipype import config
config.enable_provenance()

from nipype import Workflow, Node, MapNode, Function
from nipype.interfaces.fsl import BET, FAST, FIRST, Reorient2Std, ImageMaths, ImageStats
from nipype.interfaces.io import DataSink


def toJSON(stats, seg_file, structure_map):
    """Combine stats files to a single JSON file"""
    import json
    import os
    import nibabel as nb
    import numpy as np
    img = nb.load(seg_file)
    data = img.get_fdata()
    data = data.astype(int) # should be int
    voxel2vol = np.prod(img.header.get_zooms())
    idx = np.unique(data)
    reverse_map = {k:v for v, k in structure_map}
    out_dict = dict(zip([reverse_map[val] for val in idx], np.bincount(data.flatten())[idx]))
    for key in out_dict.keys():
        out_dict[key] = [int(out_dict[key]), voxel2vol * out_dict[key]]
    mapper = dict([(0, 'csf'), (1, 'gray'), (2, 'white')])
    out_dict.update(**{mapper[idx]: val for idx, val in enumerate(stats)})
    out_file = 'segstats.json'
    with open(out_file, 'wt') as fp:
        json.dump(out_dict, fp, sort_keys=True, indent=4, separators=(',', ': '))
    return os.path.abspath(out_file)


def create_workflow(subject_id, outdir, input_file):
    """Create a workflow for a single participant"""

    sink_directory = os.path.join(outdir, subject_id)
    
    wf = Workflow(name=f"Workflow_{subject_id}")

    orienter = Node(Reorient2Std(), name='reorient_brain')
    #wf.connect(getter, 'localfile', orienter, 'in_file')
    orienter.inputs.in_file = input_file

    better = Node(BET(), name='extract_brain')
    wf.connect(orienter, 'out_file', better, 'in_file')

    faster = Node(FAST(), name='segment_brain')
    wf.connect(better, 'out_file', faster, 'in_files')

    firster = Node(FIRST(), name='parcellate_brain')
    structures = ['L_Hipp', 'R_Hipp',
                  'L_Accu', 'R_Accu',
                  'L_Amyg', 'R_Amyg',
                  'L_Caud', 'R_Caud',
                  'L_Pall', 'R_Pall',
                  'L_Puta', 'R_Puta',
                  'L_Thal', 'R_Thal']
    firster.inputs.list_of_specific_structures = structures
    wf.connect(orienter, 'out_file', firster, 'in_file')

    fslstatser = MapNode(ImageStats(), iterfield=['op_string'], name="compute_segment_stats")
    fslstatser.inputs.op_string = ['-l {thr1} -u {thr2} -v'.format(thr1=val + 0.5, thr2=val + 1.5) for val in range(3)]
    wf.connect(faster, 'partial_volume_map', fslstatser, 'in_file')

    jsonfiler = Node(Function(input_names=['stats', 'seg_file', 'structure_map', 'struct_file'], 
                              output_names=['out_file'],
                              function=toJSON), name='save_json')
    structure_map = [('Background', 0),
                     ('Left-Thalamus-Proper', 10),
                     ('Left-Caudate', 11),
                     ('Left-Putamen', 12),
                     ('Left-Pallidum', 13),
                     ('Left-Hippocampus', 17),
                     ('Left-Amygdala', 18),
                     ('Left-Accumbens-area', 26),
                     ('Right-Thalamus-Proper', 49),
                     ('Right-Caudate', 50),
                     ('Right-Putamen', 51),
                     ('Right-Pallidum', 52),
                     ('Right-Hippocampus', 53),
                     ('Right-Amygdala', 54),
                     ('Right-Accumbens-area', 58)]
    jsonfiler.inputs.structure_map = structure_map
    wf.connect(fslstatser, 'out_stat', jsonfiler, 'stats')
    wf.connect(firster, 'segmentation_file', jsonfiler, 'seg_file')

    sinker = Node(DataSink(), name='store_results')
    sinker.inputs.base_directory = sink_directory
    wf.connect(better, 'out_file', sinker, 'brain')
    wf.connect(faster, 'bias_field', sinker, 'segs.@bias_field')
    wf.connect(faster, 'partial_volume_files', sinker, 'segs.@partial_files')
    wf.connect(faster, 'partial_volume_map', sinker, 'segs.@partial_map')
    wf.connect(faster, 'probability_maps', sinker, 'segs.@prob_maps')
    wf.connect(faster, 'restored_image', sinker, 'segs.@restored')
    wf.connect(faster, 'tissue_class_files', sinker, 'segs.@tissue_files')
    wf.connect(faster, 'tissue_class_map', sinker, 'segs.@tissue_map')
    wf.connect(firster, 'bvars', sinker, 'parcels.@bvars')
    wf.connect(firster, 'original_segmentations', sinker, 'parcels.@origsegs')
    wf.connect(firster, 'segmentation_file', sinker, 'parcels.@segfile')
    wf.connect(firster, 'vtk_surfaces', sinker, 'parcels.@vtk')
    wf.connect(jsonfiler, 'out_file', sinker, '@stats')

    return wf

def execute_workflow(workflow, working_dir, plugin, plugin_args=None):
    """Run the workflow with the specified plugin and arguments"""
    workflow.base_dir = working_dir
    workflow.config['execution']['remove_unnecessary_files'] = False
    workflow.config['execution']['poll_sleep_duration'] = 2
    workflow.config['execution']['crashdump_dir'] = working_dir
    if plugin_args:
        workflow.run(plugin, plugin_args=eval(plugin_args))
    else:
        workflow.run(plugin)
    print("Simple Workflow finished!!!")
