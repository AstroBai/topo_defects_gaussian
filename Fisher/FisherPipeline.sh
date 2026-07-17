#!/bin/bash

conda init
conda activate cosmology
cd /home/jbai/topo_defects_gaussian

cd Fisher
python GenData.py