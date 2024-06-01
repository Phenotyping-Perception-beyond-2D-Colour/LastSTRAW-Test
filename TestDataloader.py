from LastSTRAW import LastStrawData
import open3d as o3d
import numpy as np

# Global setup

# URL to download the LastSTRAW dataset
URL = "https://lcas.lincoln.ac.uk/nextcloud/index.php/s/omQY9ciP3Wr43GH/download"
# Folder to download to (must exist already)
FOLDER = "/tmp/"
# Folder created when extracting data (checked to see if extraction is required)
CHECK_FOLDER = "LAST-Straw/"
# Filename to save as 
DOWNLOAD_FILE = "download.zip"
# Default is 0 for no down sampling otherwise down sample to this voxel size
VOXEL_SIZE = 0
# If this is a path then use this to point to data files - if not None no downloading
#DATA_DIR = None
DATA_DIR = '/home/andy/Documents/CDT summer school/LastSTRAW-Test/Resources/TestData/'

from open3d._build_config import _build_config

def main():

    '''Example usage of LastStrawData importer'''

    lastStraw = LastStrawData(DATA_DIR,
                                data_dir = DATA_DIR,
                                down_sample = VOXEL_SIZE,
                                url = URL,
                                folder = FOLDER,
                                check_folder = CHECK_FOLDER,
                                download_file = DOWNLOAD_FILE)
    
    # Example of a 3D Strawberry scan in raw data
    pc, rgb, labels, fileName = lastStraw[0]
    #print(labels, fileName)
    #lastStraw.visualise(0)
    
    # Load each scan and segment by class
    classColours = {
        1: {'name': 'Leaf', 'colour': [0,0,255]},
        2: {'name': 'Stem', 'colour': [0,255, 0]},
        3: {'name': 'Fruit', 'colour': [255, 0, 255]},
        4: {'name': 'Flower', 'colour': [255, 255, 0]},
        5: {'name': 'Crown', 'colour': [255, 0, 0]},
        6: {'name': 'Background',  'colour': [255,255,255]}, # Grow bag and stray leaves
        7: {'name': 'Other part',  'colour': [255, 128, 0]},
        8: {'name': 'Platform',  'colour': [255,255,255]},
        9: {'name': 'Imature leaf', 'colour': [0,0,128]}
        }
    
    # Filter scan and segment
    for pc, rgb, labels, _ in lastStraw:
        new_PC = []
        new_RGB = []
        new_LAB = []
        for i, (pc1, rgb1, labels1) in enumerate(zip(pc,rgb,labels)):
            rgb[i] = classColours[labels1[0]]['colour']
            if not np.array_equal(rgb[i],np.array([255.,255.,255.])):
                new_PC.append(pc[i])
                new_RGB.append(rgb[i])
                new_LAB.append(labels[i])

        pointC = o3d.geometry.PointCloud()
        pointC.points = o3d.utility.Vector3dVector(new_PC)
        pointC.colors = o3d.utility.Vector3dVector(new_RGB)
        lastStraw.visualise(pointC)


if __name__ == '__main__':
    main()


        