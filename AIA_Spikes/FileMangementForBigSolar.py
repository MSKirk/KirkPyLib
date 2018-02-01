import tarfile
import fnmatch
import os
from os import listdir

'''
bin bash
cd /Volumes/BigSolar/AIA_Spikes
find . -type d >dirs.txt

cd /Volumes/BigSolar/Filtered_Spikes
xargs mkdir -p </Volumes/BigSolar/AIA_Spikes/CleanDirs.txt
'''

def directory_setup():
    fdirs = open('/Volumes/BigSolar/AIA_Spikes/dirs.txt', 'r').readlines()
    dirlist = [ss.split('/H')[0] for ss in fdirs]
    dirlist = [ss.replace('\n', '') for ss in dirlist]
    dirlist = sorted(list(set(dirlist)))

    with open("/Volumes/BigSolar/AIA_Spikes/CleanDirs.txt", "w") as CleanDirs:
        for dd in dirlist:
            CleanDirs.write("%s\n" % dd)


def spikes_unzip(year=2010):
    year = str(year)

    for file in listdir('/Volumes/BigSolar/AIA_Spikes/spikes-tars/'):
        if fnmatch.fnmatch(file, '*.tar'):
            if fnmatch.fnmatch(file, '*'+year+'*'):
                save_dir = '/Volumes/BigSolar/AIA_Spikes/'+'/'.join(file.split('_')[1].split('.')[0].split('-'))
                tar_ref = tarfile.open('/Volumes/BigSolar/AIA_Spikes/spikes-tars/'+file, 'r')
                tar_ref.extractall(save_dir)
                tar_ref.close()
                print('/'.join(file.split('_')[1].split('.')[0].split('-')))
                os.remove('/Volumes/BigSolar/AIA_Spikes/spikes-tars/'+file)

def reg_spikes_move():
    rootdir='/Volumes/BigSolar/AIA_Spikes/'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if 'H' in subdir:
                os.rename(os.path.join(subdir, file), os.path.join(subdir.split('H')[0], file))

