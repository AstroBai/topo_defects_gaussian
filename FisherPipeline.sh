#!/bin/bash

conda init
conda activate cosmology
cd /home/jbai/topo_defects_gaussian

python MakeMock.py
cd Mock
# python GetNNs.py
python MakeCovariance.py
cd ../Fisher
python GenData.py
python PlotDerivs.py
python FisherForecast.py