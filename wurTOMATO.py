################################################################
# Author     : Andy Perrett                                    #
# Contact    : aperrett@lincoln.ac.uk                          #
#            : andy@wired-wrong.co.uk                          #
# Date       : 30-05-2024                                      #
# Description: Written for use as part of the 2024 CDT summer  #
#            : school. Theme 6 - Phenotyping and beyond colour #
################################################################

# Usage: Run python from command line
#      : from LastSTRAW import LastStrawData
#      : help(LastStrawData) 

import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from tqdm import tqdm
# import requests
# from zipfile import ZipFile
from pathlib import Path
import pandas as pd

semantic_id2rgb_colour = {
    1:[255, 50, 50],
    2:[255, 225, 50],
    3:[109, 255, 50],
    4:[50, 167, 255],
                          }
# Find the maximum semantic ID to determine the size of the array
max_id = max(semantic_id2rgb_colour.keys())
# Create an array where index corresponds to the semantic ID
rgb_array = np.zeros((max_id + 1, 3), dtype=np.uint8)
# Populate the array with the RGB values
for key, value in semantic_id2rgb_colour.items():
    rgb_array[key] = value

class WurTomatoData(Dataset):
    '''
    LastStrawData inherits from Pytorch dataset class.
    LastStrawData(path: str | None [, argument=value (optional)])
        
        Arguments    : Values / types
        -----------------------------
        down_sample  : float | 0 (default: 0)
        url          : str
        folder       : str
        download_file: str
        check_folder : str

    Description:
    The LastStrawData class imports the LastSTRAW data either from
    a URL or from a given path (folder). It can also visualise the
    point cloud data using open3D. If a path is given it should
    point directly to numpy xyz files. A list of these files is
    generated in the order governed by the operating system call
    (import os) os.listdir(path).
    
    The file format is:
        X Y Z R G B class instance (each field separated by space)
        X,Y and Z: signed float
        R,G and B: unsigned int8
        class    : int64 
        instance : int64
        comments : lines starting with // (such as a header line)
    
    This class is based upon, but extensively modified, a code
    example given by LastSTRAW at https://lcas.github.io/LAST-Straw/

    Author     : Andy Perrett
    Contact    : aperrett@lincoln.ac.uk
               : andy@wired-wrong.co.uk
    Date       : 30-05-2024
    Description: Written for use as part of the 2024 CDT summer
               : school. Theme 6 - Phenotyping and beyond colour            
    
    Example usage:

    from LastSTRAW import LastStrawData
    
    URL = \"https://lcas.lincoln.ac.uk/nextcloud/index.php/s/omQY9ciP3Wr43GH/download\"
    FOLDER = \"/tmp/\"
    CHECK_FOLDER = \"LAST-Straw/\"
    DOWNLOAD_FILE = \"download.zip\"
    VOXEL_SIZE = 0
    DATA_DIR = None
    #DATA_DIR = \'/home/andy/Documents/CDT summer school/LAST-Straw/LAST-Straw/\'
    
    def main():

        lastStraw = LastStrawData(data_dir=DATA_DIR,
                                    down_sample=VOXEL_SIZE,
                                    url = URL,
                                    folder=FOLDER,
                                    check_folder = CHECK_FOLDER,
                                    download_file=DOWNLOAD_FILE)
        
        pc, rgb, labels = lastStraw[0]

        lastStraw.visualise(0)

        # Load each scan
        # for pc, rgb, _ in lastStraw:
        #     pointC = o3d.geometry.PointCloud()
        #     pointC.points = o3d.utility.Vector3dVector(pc)
        #     pointC.colors = o3d.utility.Vector3dVector(rgb)
        #     lastStraw.visualise(pointC)

    if __name__ == \'__main__\':
        main()           
    '''
    
    def __init__(self, path=Path("example_data"), **kwargs):
        
        # Default values for parameters
        self.downSample = 0
        # self.url = None
        # self.folder = None
        # self.checkFolder = None
        # self.downloadFile = None

        # # If no parameters given - do nothing more - silently
        # if path == None and len(kwargs) == 0:
        #     return

        # # Get config
        for a in kwargs:
            if a == "down_sample":
                self.downSample = kwargs[a]
        #     if a == "url":
        #         self.url = kwargs[a]
        #     if a == "folder":
        #         self.folder = kwargs[a]
        #     if a == "download_file":
        #         self.downloadFile = kwargs[a]
        #     if a == "check_folder":
        #         self.checkFolder = kwargs[a]

        # If no path given download lastStraw, unzip and continue
        if path == None:
            self.__download()
            self.__unzip()
            self.path = self.folder + self.checkFolder
        else:
            self.path = path

        # Store all numpy xyz files
        all_files = sorted(os.listdir(self.path))
        self.scans = [fname for fname in all_files if fname.endswith(".csv")]
        
    # # Download LastSTRAW data file in zip format
    # def __download(self):
    #     '''
    #     If the unzipped files exist do not download. If they do not
    #     exist then download the zip file
    #     '''
    #     print(self.folder + self.checkFolder)
    #     if not os.path.isdir(self.folder + self.checkFolder):
    #         if not os.path.isfile(self.folder + self.downloadFile):
    #             print("Downloading: " + self.downloadFile + " to folder: " + self.folder + " from: " + self.url)
    #             print("This may take a while (LastSTRAW is 4GB)...")
    #             response = requests.get(self.url, stream=True)
    #             with open(self.folder + self.downloadFile, "wb") as handle:
    #                 for data in tqdm(response.iter_content()):
    #                     handle.write(data)
    #     else:
    #         print("File already download and extracted.")


    # # Taken from https://www.geeksforgeeks.org/unzipping-files-in-python/
    # def __unzip(self):
    #     '''
    #     If data zip file has been download, extract all files
    #     and delete downloaded zip file
    #     '''
    #     if os.path.isfile(self.folder + self.downloadFile): 
    #         if not os.path.isdir(self.folder+ self.checkFolder):
    #             print("Extracting: " + self.folder + self.downloadFile)
    #             with ZipFile(self.folder + self.downloadFile, 'r') as zObject: 
    #                 zObject.extractall(path=self.folder) 
    #             print("Deleting " + self.folder + self.downloadFile)
    #             os.remove(self.folder + self.downloadFile)

    # Loads point cloud data files
    def __load_as_array(self, index):
        # Loads the data from an .xyz file into a numpy array.
        # Also returns a boolean indicating whether per-point labels are available.
        # data_array = np.loadtxt(self.path / self.scans[index]) # raw data as np array, of shape (nx6) or (nx8) if labels are available.
        data_array = pd.read_csv(self.path / self.scans[index])

        labels_available = data_array.shape[1] == 19
        return data_array, labels_available

    # Loads point cloud from file in Numpy array. Returns point cloud
    def __load_as_o3d_cloud(self, index):
        # Loads the data from an .xyz file into an open3d point cloud object.
        data, labels_available = self.__load_as_array(index)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data[["x", "y", "z"]].values)
        pc.colors = o3d.utility.Vector3dVector(data[["red", "green", "blue"]].values)
        labels = None
        if labels_available:
            labels = data[["semantic", "leaf_stem_instances", "leaf_instances", "stem_instances"]].values
        # if load_skeleton_data:
            skeleton_data = data.loc[~data["x_skeleton"].isna(), ["x_skeleton",	"y_skeleton", "z_skeleton",	"vid",	"parentid",	"edgetype"]]
        return pc, labels_available, labels, skeleton_data

    # Saves the point cloud data - TODO NOTE untested
    # def save_data_as_xyz(self, data, fileName):
    #     # To save your own data in the same format as we used, you can use this function.
    #     # Edit as needed with more or fewer columns.
    #     with open(self.path + fileName, 'w') as f:
    #         f.write("//X Y Z R G B class instance\n")
    #         np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d'])
    #     return

    # Invokes Open3D to visualise point cloud
    def visualise(self, i):
        if isinstance(i, int):
            name = self.scans[i]
            pointCloud,_,_ = self.__load_as_o3d_cloud(i)
        else:
            name = "PointCloud"
            pointCloud = i
        pointCloud.colors = o3d.utility.Vector3dVector(np.asarray(pointCloud.colors) / 255)
        if self.downSample != 0:
            pointCloud = pointCloud.voxel_down_sample(voxel_size=self.downSample)
        vis = o3d.visualization.draw_geometries([pointCloud], window_name=name)
        del vis

    def visualise_semantic(self, i):
        if isinstance(i, int):
            name = self.scans[i]
            pointCloud,_,labels = self.__load_as_o3d_cloud(i)
        else:
            name = "PointCloud"
            pointCloud = i
        colors = rgb_array[labels[:,0].astype(int)]
        pointCloud.colors = o3d.utility.Vector3dVector(colors / 255)
        if self.downSample != 0:
            pointCloud = pointCloud.voxel_down_sample(voxel_size=self.downSample)
        vis = o3d.visualization.draw_geometries([pointCloud], window_name=name)
        del vis

    def visualize_skeleton(self, i, parent_nodes_only=True):
        pointCloud,_,labels, skeleton_data = self.__load_as_o3d_cloud(i)
        pointCloud.colors = o3d.utility.Vector3dVector(np.asarray(pointCloud.colors) / 255)

        pc_skelet = o3d.geometry.PointCloud()

        if parent_nodes_only:
            parent_nodes_only = self.get_only_parent_nodes(skeleton_data)
            colors = np.repeat(np.array([[1, 0., 0.]]), len(parent_nodes_only), axis=0)
            pc_skelet.points = o3d.utility.Vector3dVector(parent_nodes_only)
            pc_skelet.colors = o3d.utility.Vector3dVector(colors)
        else:
            pc_skelet.points = o3d.utility.Vector3dVector(skeleton_data[["x_skeleton",	"y_skeleton", "z_skeleton"]].values)
            colors = np.repeat(np.array([[0.5, 0.5, 0.5]]), len(skeleton_data), axis=0)

        vis = o3d.visualization.draw_geometries([pc_skelet], window_name="PointCloud")
        ## 
        pd.DataFrame(parent_nodes_only).to_csv("debug.txt", index=False)
        print("x")

    def get_only_parent_nodes(self, skeleton_data):
        parentids = skeleton_data.loc[skeleton_data["edgetype"]=="+", "parentid"].values
        parent_nodes_only = skeleton_data[skeleton_data['vid'].isin(parentids)][["x_skeleton", "y_skeleton", "z_skeleton"]].values
        return parent_nodes_only

    # Return number of data files
    def __len__(self):
        return len(self.scans)
    
    # Returns 3 Numpy arrays [[x,y,z]], [[R,G,B]] and [[labels]]
    def __getitem__(self, index):
        pointCloud, labels_available, labels, skeleton_data = self.__load_as_o3d_cloud(index)
        if self.downSample != 0:
            pointCloud = pointCloud.voxel_down_sample(voxel_size=self.downSample)
        pc = pointCloud.points
        rgb = pointCloud.colors

        return np.asarray(pc), np.asarray(rgb), np.asarray(labels), skeleton_data


if __name__=="__main__":
    obj = WurTomatoData()
    pc, rgb, labels, _ = obj[0]
    # obj.visualise(0)
    # obj.visualise_semantic(0)
    obj.visualize_skeleton(0)

