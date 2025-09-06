What is input and output of this competition?
INPUT:
Brain imaging series from multiple modalities like CTA , MRA and MRI , T1 post contrast and T2 weighted.
Each row of tabular labels correspond to one image series identified by series id
OUTPUT:
A binary label Aneurysm Present.
13 additional binary targets for presence at specific anatomic locations.
A localization file with spatial annotations (coordinates and an approximate size/radius)
 for each aneurysm.
So like for each image we should output 14 probabitlies 
