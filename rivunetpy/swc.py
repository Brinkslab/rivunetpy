import copy
import math
import numpy as np
import time
from collections import Counter
from random import gauss
from random import random, randrange
from itertools import cycle

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)

from rivunetpy.utils.io import saveswc, loadswc
from rivunetpy.soma import Soma
from rivunetpy.utils.metrics import euclidean_distance
from rivunetpy.utils.color import RGB_from_hex

LABELS = {-1 : 'Root',
          0  : 'Undefined',
          1  : 'Soma',
          2  : 'Axon',
          3  : '(Basal) Dendrite',
          4  : 'Apical Dentrite',
          5  : 'Branch Point',
          6  : 'End Point',
          7  : 'Other'}

COLORS = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])


class SWC(object):

    def __init__(self, soma=None):
        self._data = np.zeros((1, 8))

        ########## PLOTTING PARAMETERS ##########
        self.swc_density = 90
        self.swc_fancy = False
        #########################################

        if soma:
            self._data[0, :] = np.asarray([0, 1, soma.centroid[0], soma.centroid[
                1], soma.centroid[2], soma.radius, -1, 1])

    def add(self, swc_nodes):
        np.vstack((self._data, swc_nodes))

    def add_branch(self, branch, pidx=None, random_color=False, label=False):
        '''
        Add a branch to swc.
        Note: This swc is special with N X 8 shape. The 8-th column is the online confidence
        '''
        if random_color:
            rand_node_type = randrange(256)

        new_branch = np.zeros((len(branch.pts), 8))
        id_start = 1 if self._data.shape[
                            0] == 1 else self._data[:, 0].max() + 1

        for i in range(len(branch.pts)):
            p, r, c = branch.pts[i], branch.radius[i], branch.conf[i]
            id = id_start + i
            # 3 for basal dendrite; 4 for apical dendrite;
            # However now we cannot differentiate them automatically
            nodetype = 3

            if i == len(branch.pts) - 1:  # The end of this branch
                pid = self._data[pidx, 0] if pidx is not None else -2
                if pid != -2 and pid != 0 and self._data.shape[0] != 1:
                    # Its connected node is fork point
                    self._data[self._data[:, 0] == pid, 1] = 5 if label else 3
            else:
                pid = id_start + i + 1
                if i == 0:
                    nodetype = 6 if label else 3 # Endpoint

            assert (pid != id)
            new_branch[i] = np.asarray([
                id, rand_node_type
                if random_color else nodetype, p[0], p[1], p[2], r, pid, c])

        # Check if any tail should be connected to its tail
        tail = new_branch[0]
        matched, minidx = self.match(tail[2:5], tail[5])
        if matched and self._data[minidx, 6] == -2:
            self._data[minidx, 6] = tail[0]

        self._data = np.vstack((self._data, new_branch))

    def clean(self):

        sec_list = [1, 2, 3, 4] # Valid TypeIDs for NetPYne
        nanint = np.zeros(1)
        nanint[0] = np.nan
        nanint = nanint.astype(int)[0]

        SampleIDs = self._data[:, 0].astype(int)  # SampleID

        swc_dict, swc_children, swc_ends, swc_indices = self.swc_to_dicts()

        # Map from old SampleIDs to NewSampleIDs. Index is old SampleID
        mapper = np.full(np.amax(SampleIDs)+1, np.NaN, dtype=int)

        if -1 in swc_children.keys():
            ROOT_ID = swc_children[-1][0] # ParentID of root is -1
        elif any([key in value for key, value in swc_children.items()]): # Search for attractors
            ii = np.argmax([key in value for key, value in swc_children.items()])
            ROOT_ID = int(list(swc_children.keys())[ii])
        else:
            ROOT_ID = 0

        ID = 0 # Start ID

        # Delete non-root attractors
        root_index = np.argmax(SampleIDs == ROOT_ID)
        attractor_ii = []
        for ii, (SampleID, ParentID) in enumerate(self._data[:, [0, 6]]):
            if SampleID == ParentID:
                attractor_ii.append(ii)
        attractor_ii = np.array(attractor_ii)
        attractor_ii = attractor_ii[attractor_ii != root_index]
        for ii in attractor_ii:
            self._data = np.delete(self._data, ii, axis=0)

        def assign_ID(current_node):
            nonlocal ID
            nonlocal mapper
            nonlocal swc_ends

            ID += 1
            new_ID = ID

            mapper[current_node] = new_ID


            if current_node in swc_ends:
                pass
            else:
                for child_node in swc_children[current_node]:
                    if child_node == current_node:
                        continue  # Avoid attractors, e.g. the root
                    else:
                        assign_ID(child_node)

        assign_ID(ROOT_ID)

        assert (np.unique(mapper[mapper != nanint])
                == np.sort(mapper[mapper != nanint])).all(), 'Incorrect mapper, doubly mapped indecies.'

        # index_mapper = np.full_like(mapper, np.NaN)
        # for index, SampleID in enumerate(SampleIDs):
        #     index_mapper[SampleID] = index

        new_data = np.zeros((np.amax(mapper), self._data.shape[1]))
        for data_line, old_SampleID in zip(self._data, SampleIDs):

            new_SampleID = mapper[old_SampleID]
            new_ParentID = mapper[int(data_line[6])]

            if new_SampleID == 473.0:
                pass

            new_data[new_SampleID - 1, :] = data_line
            new_data[new_SampleID - 1, 0] = new_SampleID
            new_data[new_SampleID - 1, 6] = new_ParentID

        new_data[np.logical_not(np.isin(new_data[:, 1], sec_list)), 1] = 3 # All non-valid TypeIDs are converted to 3 (dendrite)



        self._data = new_data

        # Set ParentID of point 0 to -1 (Root).
        self._data[0, 6] = -1

        check = (self._data[:, 0] > self._data[:, 6]).all()
        if not check:
            ii = np.argmax(self._data[:, 0] <= self._data[:, 6])
            raise AssertionError(f'At index {ii}, SampleID is {self._data[ii, 0]}, while '
                                 f'ParentID is {self._data[ii, 6]}')

    def apply_soma_TypeID(self, soma: Soma):

        mask = soma.mask.flatten()
        indices = np.ravel_multi_index(self._data[:, [2, 3, 4]].astype(int).T, soma.mask.shape)
        for ii, pos in enumerate(indices):
            if mask[pos]:
                # If in mask apply ID
                self._data[ii, 1] = 1

        # Reset ParentID of point 0 to -1 (Root), as this might be changed by the mask
        self._data[0, 6] = -1

    def apply_scale(self, factors: tuple):

        self._data[:, 2] = self._data[:, 2] * factors[0] # Scale X
        self._data[:, 3] = self._data[:, 3] * factors[1] # Scale Y
        self._data[:, 4] = self._data[:, 4] * factors[2] # Scale Y

        # TODO: Nice implementation for scaling section radii based on voxel size
        # This would likely involve figuring out where the line spanning from childID to parentID is pointing to
        # correctly scale the radius with an appropriate weight for each direction
        self._data[:, 5] = self._data[:, 5] * np.mean(factors)


    def _prune_leaves(self):
        # Find all the leaves
        childctr = Counter(self._data[:, 6])
        leafidlist = [id for id in self._data[:, 0]
                      if id not in self._data[:, 6]]
        id2dump = []
        rmean = self._data[:, 5].mean()  # Mean radius

        for leafid in leafidlist:  # Iterate each leaf node
            nodeid = leafid
            branch = []
            while True:  # Get the leaf branch out
                node = self._data[self._data[:, 0] == nodeid, :].flatten()
                if node.size == 0:
                    break
                branch.append(node)
                parentid = node[6]
                if childctr[parentid] != 1:
                    break  # merged / unconnected
                nodeid = parentid

            # Get the length of the leaf
            leaflen = sum([
                np.linalg.norm(branch[i][2:5] - branch[i - 1][2:5])
                for i in range(1, len(branch))
            ])

            # Prune if the leave is too short or
            # the confidence of the leave branch is too low
            if leaflen <= 4 * rmean:
                id2dump.extend([node[0] for node in branch])

        # Only keep the swc nodes not in the dump id list
        cutted = []
        for nodeidx in range(self._data.shape[0]):
            if self._data[nodeidx, 0] not in id2dump:
                cutted.append(self._data[nodeidx, :])

        cutted = np.squeeze(np.dstack(cutted)).T
        self._data = cutted

    def _prune_unreached(self):
        '''
        Only keep the largest connected component
        '''
        swcdict = {}
        for n in self._data:  # Hash all the swc nodes
            swcdict[n[0]] = Node(n[0])

        # Try to join all the unconnected branches at first
        for i, n in enumerate(self._data):
            if n[6] not in swcdict:
                # Try to match it
                matched, midx = self.match(n[2:5], n[5])
                if matched:
                    self._data[i, 6] = self._data[midx, 0]

        # Add mutual links for all nodes
        for n in self._data:
            id = n[0]
            pid = n[6]
            if pid >= 0:
                swcdict[id].add_link(swcdict[pid])

        groups = connected_components(set(swcdict.values()))
        lenlist = [len(g) for g in groups]
        maxidx = lenlist.index(max(lenlist))
        set2keep = groups[maxidx]
        id2keep = [n.id for n in set2keep]
        self._data = self._data[
                     np.in1d(self._data[:, 0], np.asarray(id2keep)), :]

    def prune(self):
        self._prune_unreached()
        self._prune_leaves()

    def reset(self, crop_region, zoom_factor):
        '''
        Pad and rescale swc back to the original space
        '''

        tswc = self._data.copy()
        if zoom_factor != 1.:  # Pad the swc back to original space
            tswc[:, 2:5] *= 1. / zoom_factor

        # Pad the swc back
        tswc[:, 2] += crop_region[0, 0]
        tswc[:, 3] += crop_region[1, 0]
        tswc[:, 4] += crop_region[2, 0]
        self._data = tswc

    def get_id(self, idx):
        return self._data[idx, 0]

    def match(self, pos, radius):
        '''
        Find the closest ground truth node 
        '''

        nodes = self._data[:, 2:5]
        distlist = np.squeeze(cdist(pos.reshape(1, 3), nodes))
        if distlist.size == 0:
            return False, -2
        minidx = distlist.argmin()
        minnode = self._data[minidx, 2:5]

        # See if either of them can cover each other with a ball of their own
        # radius
        mindist = np.linalg.norm(pos - minnode)
        return radius > mindist or self._data[minidx, 5] > mindist, minidx

    def size(self):
        return self._data.shape[0]

    def save(self, fname):
        saveswc(fname, self._data)

    def set_view_density(self, perc):
        assert 1 <= perc <= 100, 'Quantile of segments (in %) to plot should be between 1 and 100'
        self.swc_density = perc

    def set_fanciness(self, fancy):
        self.swc_fancy = fancy

    def get_array(self):
        return self._data[:, :7]

    def view(self):
        from rivunetpy.utils.rendering3 import Viewer3, Line3

        # Compute the center of mass
        center = self._data[:, 2:5].mean(axis=0)
        translated = self._data[:, 2:5] - \
                     np.tile(center, (self._data.shape[0], 1))

        # Init viewer
        viewer = Viewer3(800, 800, 800)
        viewer.set_bounds(self._data[:, 2].min(), self._data[:, 2].max(),
                          self._data[:, 3].min(), self._data[:, 3].max(),
                          self._data[:, 4].min(), self._data[:, 4].max())
        lid = self._data[:, 0]

        line_color = [random(), random(), random()]
        for i in range(self._data.shape[0]):
            # Change color if its a bifurcation
            if (self._data[i, 0] == self._data[:, -1]).sum() > 1:
                line_color = [random(), random(), random()]

            # Draw a line between this node and its parent
            if i < self._data.shape[0] - 1 and self._data[i, 0] == self._data[i + 1, -1]:
                l = Line3(translated[i, :], translated[i + 1, :])
                l.set_color(*line_color)
                viewer.add_geom(l)
            else:
                pid = self._data[i, -1]
                pidx = np.argwhere(pid == lid).flatten()
                if len(pidx) == 1:
                    l = Line3(translated[i, :], translated[pidx, :].flatten())
                    l.set_color(*line_color)
                    viewer.add_geom(l)

        while (True):
            try:
                viewer.render(return_rgb_array=False)
            except KeyboardInterrupt:
                break

    def swc_to_dicts(self):
        # Create connectivity dictionary

        swc_dict = {}
        swc_children = {}
        swc_indices = {}
        for ii, line in enumerate(copy.copy(self._data).astype(int)):
            SampleID, ParentID = (line[0], line[6])

            swc_dict[SampleID] = ParentID
            swc_indices[SampleID] = ii

            # As we're iterating over the SWC anyways, we might as well
            # retrieve the children too.
            if ParentID not in swc_children:
                swc_children[ParentID] = []

            swc_children[ParentID].append(SampleID)

        swc_ends = []
        for SampleID, ParentID in swc_dict.items():
            if SampleID not in swc_children:
                swc_ends.append(int(SampleID))

        return swc_dict, swc_children, swc_ends, swc_indices

    def get_all_segments(self):
        swc_dict, swc_children, swc_ends, swc_indices = self.swc_to_dicts()

        segment_maps = []
        for end in swc_ends:
            segment = [swc_indices[end]]
            SampleID = end
            ParentID = swc_dict[SampleID]

            while (SampleID != ParentID) and (ParentID in swc_dict):
                SampleID = swc_dict.pop(SampleID)
                ParentID = swc_dict[SampleID]
                segment.append(swc_indices[SampleID])  # Segments consist of INDICES in original SWC

            # segment.append(swc_indices[ParentID])
            segment_maps.append(segment)

        return segment_maps

    def as_actor(self, color=None, centered=False):
        # Create the polydata where we will store all the geometric data
        # https://stackoverflow.com/questions/17547851/create-vtkpolydata-object-from-list-with-tuples-in-python
        # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/LongLine

        if color is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = cycle(prop_cycle.by_key()['color'])
            color = next(colors)

        # Compute the center of mass
        center = self._data[:, 2:5].mean(axis=0)

        # Compute relative coordinates
        if centered:
            coords = self._data[:, 2:5] - \
                     np.tile(center, (self._data.shape[0], 1))
        else:
            coords = self._data[:, 2:5]

        # Get radii
        radii = self._data[:, 5]

        # Get Sample IDs
        sample_ids = self._data[:, 0]

        # Empty array of actors to be filled and converted into an assembly later
        actors = []
        assembly = vtk.vtkAssembly()

        if not self.swc_fancy:
            # Get the dictionaries outlining the connectivities. From these, compute the indecies that map to segments
            segment_maps = self.get_all_segments()
            segment_points = []
            for map in segment_maps:
                segment_points.append(self._data[map, 2:5])

            # Using the index map, plot each branch:
            for point_set in segment_points:
                points = vtkPoints()

                for point in point_set:
                    point = (point[0], point[1], point[2])
                    points.InsertNextPoint(point)

                lines = vtkCellArray()
                for ii in range(len(point_set) - 1):
                    line = vtkLine()
                    line.GetPointIds().SetId(0, ii)
                    line.GetPointIds().SetId(1, ii + 1)
                    lines.InsertNextCell(line)

                # Create a polydata to store everything in
                linesPolyData = vtkPolyData()

                # Add the points to the dataset
                linesPolyData.SetPoints(points)

                # Add the lines to the dataset
                linesPolyData.SetLines(lines)

                # Setup color, actor and mapper
                colors = vtkNamedColors()
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(linesPolyData)

                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetLineWidth(2)
                actor.GetProperty().SetColor(RGB_from_hex(color))

                actors.append(actor)

            for actor in actors:
                assembly.AddPart(actor)

        else:  # Fancy render
            # https://kitware.github.io/vtk-examples/site/Python/PolyData/TubeFilter/

            line_lengths = []

            for ii in range(self._data.shape[0]):

                # Change color if its a bifurcation
                if (self._data[ii, 0] == self._data[:, -1]).sum() > 1:
                    pass

                lineSource = vtkLineSource()

                # Draw a line between this node and its parent
                if ii < self._data.shape[0] - 1 and self._data[ii, 0] == self._data[ii + 1, -1]:
                    pp1 = coords[ii, :]
                    pp2 = coords[ii + 1, :]
                    lineSource.SetPoint1(*pp1)
                    lineSource.SetPoint2(*pp2)
                else:
                    parent_id = self._data[ii, -1]
                    pidx = np.argwhere(parent_id == sample_ids).flatten()
                    if len(pidx) == 1:
                        pp1 = coords[ii, :]
                        pp2 = coords[pidx, :].flatten()
                        lineSource.SetPoint1(*pp1)
                        lineSource.SetPoint2(*pp2)
                    else:
                        pp1 = None
                        pp2 = None

                if pp1 is None:  # If a line should not be drawn, contunue
                    continue

                line_lengths.append(euclidean_distance(pp1, pp2))

                # Setup actor and mapper
                lineMapper = vtkPolyDataMapper()
                lineMapper.SetInputConnection(lineSource.GetOutputPort())

                lineActor = vtkActor()
                lineActor.SetMapper(lineMapper)

                # Create tube filter
                tubeFilter = vtkTubeFilter()
                tubeFilter.SetInputConnection(lineSource.GetOutputPort())
                tubeFilter.SetRadius(radii[ii])
                tubeFilter.SetNumberOfSides(12)
                tubeFilter.CappingOn()
                tubeFilter.Update()

                # Setup actor and mapper
                tubeMapper = vtkPolyDataMapper()
                tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())

                tubeActor = vtkActor()
                tubeActor.SetMapper(tubeMapper)

                # Set transparency.
                tubeActor.GetProperty().SetOpacity(1)

                # Set color
                tubeActor.GetProperty().SetColor(RGB_from_hex(color))

                # Cap end points
                tubeActor

                actors.append(tubeActor)

            quant = np.quantile(line_lengths, (100 - self.swc_density) / 100)

            for line_length, actor in zip(line_lengths, actors):
                if line_length > quant:
                    assembly.AddPart(actor)

        assembly.SetOrigin(center)
        return assembly

    @staticmethod
    def get_TypeID_label(typeid: int):
        typeid = int(typeid)
        # Clip label between -1 and 7
        typeid = max(typeid, -1)
        typeid = min(typeid, 7)

        return LABELS[typeid]

    @staticmethod
    def get_TypeID_color(typeid: int):
        typeid = int(typeid)
        typeid = max(typeid, -1)
        typeid = min(typeid, 7)
        return COLORS[typeid + 1]

    def extents(self):
        return np.amax(self._data[:, 2:5], axis=0) - np.amin(self._data[:, 2:5], axis=0)


    def push_nodes_with_binary(self, b, step_ratio=0.1, niter=0):
        '''
        Push the nodes towards the center with the binary image boundaries
        '''
        lid = list(self._data[:, 0])
        lpid = list(self._data[:, -2])
        t_data = self._data.copy()

        children_idx = {pid: [i for i, p in enumerate(
            lpid) if p == t_data[i, 0]] for pid in lpid}

        for _ in range(niter):
            for i in range(t_data.shape[0]):
                pid, radius, (x, y, z) = int(
                    t_data[i, -2]), t_data[i, -3], t_data[i, 2:5]
                cidx = children_idx[pid]
                if pid != i and pid in lid and len(cidx) <= 1:
                    px, py, pz = t_data[t_data[:, 0] == pid, 2:5][0]
                    vnorm = norm_vec(np.asarray([x - px, y - py, z - pz]))

                    if len(cidx) == 1:
                        cx, cy, cz = t_data[cidx[0], 2:5]
                        vnorm = (
                                        vnorm + norm_vec(np.asarray([cx - x, cy - y, cz - z]))) / 2
                    if all([v == 0 for v in vnorm]):
                        continue

                    pt = np.asarray([x, y, z])
                    p_vectors = get_perpendicular_vectors(
                        pt, vnorm)
                    p_distances = [get_distance_to_boundary(
                        pt, pvec, b) for pvec in p_vectors]
                    dx, dy, dz = np.sum(
                        [pv * pd for pv, pd in zip(p_vectors, p_distances)], 0)

                    # Constrain the displacement by the nodo radii
                    tx = x + dx * step_ratio
                    ty = y + dy * step_ratio
                    tz = z + dz * step_ratio
                    dist = ((tx - self._data[i, 2]) ** 2 +
                            (ty - self._data[i, 3]) ** 2 +
                            (tz - self._data[i, 4]) ** 2) ** 0.5
                    if dist <= radius / 2:
                        t_data[i, 2] = tx
                        t_data[i, 3] = ty
                        t_data[i, 4] = tz
                else:
                    pass
        self._data = t_data


def get_distance_to_boundary(pt, vec, b):
    temp_pt = pt.copy()
    while (True):
        next_pt = temp_pt + vec
        if b[math.floor(next_pt[0]),
             math.floor(next_pt[1]),
             math.floor(next_pt[2])] <= 0:

            return ((temp_pt - pt) ** 2).sum() ** 0.5
        else:
            temp_pt = next_pt


def norm_vec(vec):
    norm = (vec ** 2).sum() ** 0.5
    return vec / norm


def get_perpendicular_vectors(pt, vec):
    v1 = perpendicular_vector(vec)
    v2 = -v1
    v3 = perpendicular_vector(vec, v1)
    v4 = -v3
    return v1, v2, v3, v4


def make_rand_vector3d():
    vec = [gauss(0, 1) for i in range(3)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def perpendicular_vector(v, vr=None):
    return np.cross(v, make_rand_vector3d() if vr is None else vr)


def get_subtree_nodeids(swc, node):
    subtreeids = np.array([])

    # Find children
    chidx = np.argwhere(node[0] == swc[:, 6])

    # Recursion stops when there this node is a
    # leaf with no children, return itself
    if chidx.size == 0:
        return node[0]
    else:
        # Get the node ids of each children
        for c in chidx:
            subids = get_subtree_nodeids(swc, swc[c, :].squeeze())
            subtreeids = np.hstack((subtreeids, subids, node[0]))

    return subtreeids


class Node(object):

    def __init__(self, id):
        self.__id = id
        self.__links = set()

    @property
    def id(self):
        return self.__id

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)


def connected_components(nodes):
    '''
    The function to look for connected components.
    Reference: https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
    '''

    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes
        # connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:
            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result




def clean(filenames):
    import os
    from rivunetpy.utils.extensions import RIVULET_2_TREE_SWC_EXT

    if type(filenames) is not list:
        filenames = [filenames]

    for filename in filenames:
        swc_mat = loadswc(filename)
        swc = SWC()
        swc._data = swc_mat

        # print(f'Cleaning {filename} ...')
        swc.clean()

        fname = os.path.splitext(filename)[0] + RIVULET_2_TREE_SWC_EXT
        swc.save(fname)