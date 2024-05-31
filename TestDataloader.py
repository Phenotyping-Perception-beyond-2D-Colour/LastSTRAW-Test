from LastSTRAW import LastStrawData
import open3d as o3d

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
    print(labels, fileName)
    lastStraw.visualise(0)

    # # Load each scan
    # for pc, rgb, labels in lastStraw:
    #     pointC = o3d.geometry.PointCloud()
    #     pointC.points = o3d.utility.Vector3dVector(pc)
    #     pointC.colors = o3d.utility.Vector3dVector(rgb)
    #     lastStraw.visualise(pointC)


if __name__ == '__main__':
    main()


        