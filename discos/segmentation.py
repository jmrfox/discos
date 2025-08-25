"""
Systematic mesh segmentation implementation using trimesh.

Important terminology:
- "Original Mesh": A mesh is a 3D object defined by a set of vertices, edges, and faces. Original means this is the mesh that is segmented. 
    For use in DISCOS, the original mesh should be watertight (closed), have no self-intersections, and be single-hull (no multiple hulls).
- "Bounding planes": The two planes of constant z-value that bound the original mesh: one at the minimum z-value and one at the maximum z-value.
- "Cut": A plane of constant z-value that intersects the mesh. A single cut always has at least one cross-section associated with it.
- "Slice": A 3D partition of the mesh with its volume either between two cuts, or between one cut and a bounding plane. 
    A single slice might contain multiple closed volumes.
- "Segment": A *contiguous* 3D volume of the mesh with its volume between two cuts. Each segment has a single 
    contiguous external surface and at least one contiguous internal surface.
    Segments are the basic building blocks of the mesh, and are the nodes of the segment graph. 
- "Cross-section": A contiguous 2D area resulting from intersecting the mesh with a cut. It has a x,y,z position and a surface area 
    (internal to the original mesh volume) in the xy-plane. 
    Cross-sections occur at the nodes of the segment graph. Every cross-section is shared by two segments.
    Every cross-section has a fitted disk with radius and center position. 
    (Several algorithms exist for fitting the disk, e.g. least squares, least absolute deviations, etc.)
- "Internal surface area": The surface area of a segment that overlaps a cross-section. 
- "External surface area": The surface area of a segment that is shared with the original mesh.
- "Skeleton": A graph representation where segments are edges and nodes are cross sections (shared by two segments in neighboring slices)
    which share a cross section. No two nodes at the same cut (z-value) may be connected by an edge.

The algorithm follows these steps:

1. Validate input mesh: Check if mesh is watertight, has no self-intersections, and is single-hull. 
    If not, raise an error. Mesh can be passed as a trimesh object or as an instance of discos.mesh.MeshManager.

2. Create cuts along z-axis using trimesh functionality. The cuts are spaced at regular intervals along the z-axis equal to the slice thickness.
   Where cuts intersect the mesh, identify the cross-sections.

3. Identify segments within each slice. Segments are identified as the contiguous volume components of the mesh within each slice.

4. Build segment graph based on shared internal faces. 

5. Validate volume and surface area conservation. The sum of the volumes of the segments should be equal to the volume of the original mesh.
    The sum of the external surface areas of the segments should be equal to the surface area of the original mesh.


EXAMPLE:
    Consider a cylinder with radius 1 m and height 4 m, aligned with the z-axis. 
    It's volume is 4*pi*1^2 = 4pi m^3, and it's surface area is 4*2pi*1 (side) + 2*pi*1^2 (caps) = 10pi m^2.
    When segmented with slice_height = 1 m, it should be segmented into 4 segments. Each segment should have the same volume = pi m^3.
    The two segments on the ends will have the same external surface area = 2*pi*1*1 (side) + pi*1^2 (cap) = 3pi m^2 
        and the same internal surface area = pi*1^2 (cross section) = pi m^2.
    The two segments in the middle will have the same external surface area = 2*pi*1*1 (side) = 2pi m^2 
        and the same internal surface area = 2*pi*1^2 (two cross sections) = 2pi m^2.



"""

import numpy as np
import trimesh
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import math
