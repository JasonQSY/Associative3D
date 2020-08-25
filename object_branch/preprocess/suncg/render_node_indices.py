#!/usr/bin/python
# Example usage python render_node_indices.py --min=1 --nc=1
import argparse
import os
import os.path as osp
import threading

import subprocess
import time

parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument('--nc', type=int, help='number of cores')
parser.add_argument('--min', type=int, help='min id')
parser.add_argument('--max', type=int, default=0, help='max id')
parser.add_argument('--mesa', type=bool, default=False, help='Use Mesa')
args = parser.parse_args()

sunCgDir = osp.join('/nfs.yoda/imisra/nileshk/', 'suncg_debug')
# sunCgDir = osp.join('/scratch/nileshk/nvme-scratch/nileshk/', 'debug_suncg')
toolboxDir = osp.join(sunCgDir, 'toolbox')
execFolder = 'gaps/bin/x86_64'

modelsAll =  [f for f in os.listdir(osp.join(sunCgDir, 'camera'))]
list.sort(modelsAll)

nCores = args.nc
nMin = args.min
nMax = args.max
if(nMax == 0):
    nMax = len(modelsAll)

class renderingThread(threading.Thread):
    def __init__(self, c):
        threading.Thread.__init__(self)
        self.c = c
        
    def run(self):
        for ix in range(nMin-1, nMax):
            if(ix % nCores == self.c):
                modelId = modelsAll[ix]
                modelDir = osp.join(sunCgDir, 'house', modelId)
                saveDir = osp.join(sunCgDir, 'renderings_node', modelId)
                camFile = osp.join(sunCgDir, 'camera', modelId, 'room_camera.txt')
                
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                renderFlags = '-capture_node_images'
                if args.mesa:
                    renderFlags += ' -mesa'
                renderCommand = 'cd {}; nice -n 20  {}/scn2img house.json {} {} {};'.format(modelDir, osp.join(toolboxDir, execFolder), renderFlags, camFile, saveDir)

                os.system(renderCommand)

tList = [renderingThread(c) for c in range(nCores)]

for renderer in tList:
    renderer.start()
    
