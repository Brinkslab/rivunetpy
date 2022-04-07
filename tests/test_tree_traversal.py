import numpy as np
# SampleID followed by ParentID
tree = [
    [0, 0],
    [1, 0],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 5],
    [7, 6],
    [8, 7],
    [9, 8],
    [10, 9],
    [11, 4],
    [12, 11],
    [13, 12],
    [14, 13],
    [15, 14],
    [16, 15],
    [17, 16],
    [18, 14],
    [19, 18],
    [20, 19],
    [21, 7],
    [22, 21]
]

import time

def find_children(tree):
    children = {}
    for SampleID, ParentID in tree:
        if ParentID not in children:
            children[ParentID] = []

        children[ParentID].append(SampleID)
    return children

def get_all_paths(node, children):
    if node not in children:
        return [[node]]
    return [
        [node] + path for child in children[node] for path in get_all_paths(child, children)
    ]




def convert_tree_to_dict(tree):
    tree_dt = {}
    for SampleID, ParentID in tree:
        tree_dt[SampleID] = ParentID
    return tree_dt

def get_all_segments(tree):
    children = find_children(tree)
    ends = find_ends(tree, children)
    tree_dt = convert_tree_to_dict(tree)

    print(f'Ends: {ends}')

    segments = []
    for end in ends:
        segment = [end]
        SampleID = end
        ParentID = tree_dt[SampleID]

        while (SampleID != ParentID) and (ParentID in tree_dt):
            SampleID = tree_dt.pop(SampleID)
            ParentID = tree_dt[SampleID]
            segment.append(SampleID)

        segment.append(ParentID)
        segments.append(segment)

    return segments


if __name__ == '__main__':
    start = time.time()
    segments = get_all_segments(tree)
    print(f'Segments: {segments}')
    print(f'Took: {(time.time()-start)*1E3} ms')