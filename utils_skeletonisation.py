import numpy as np
import pandas as pd
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from plant_registration_4d import skeleton
from plant_registration_4d import visualize as vis
from plant_registration_4d import skeleton_matching as skm


def get_only_parent_nodes(skeleton_data, max_node_order=1):
    """
    This function is a bit complitated, but what it does it that is only gets nodes that have a child,
    also known as parent nodes.
    Furthermore an additional filtering is done to only select nodes of order x.
    It returns a skeleton with updates edges indexes to be able to use the 4d_plant_registration library


    """
    parentids = skeleton_data.loc[skeleton_data["edgetype"] == "+", "parentid"].values
    vid_dict = {}
    edges_dict = {}
    for x in skeleton_data.index[::-1]:
        if x == 0 or x == 1:
            print("debug started")

        root = False
        parentid = skeleton_data.iloc[x].parentid
        counter = 0
        if skeleton_data.edgetype.iloc[x] == "+":  # to correct for itself as well
            counter += 1
        while not root and x != 0:
            # while not root:
            temp = skeleton_data[skeleton_data["vid"] == parentid]
            if temp.edgetype.iloc[0] == "+":
                counter += 1
            parentid = temp.parentid.iloc[0]

            if (
                skeleton_data.iloc[x].parentid in parentids
                and skeleton_data.iloc[x].parentid not in edges_dict.keys()
                and parentid in parentids
            ):
                edges_dict[skeleton_data.iloc[x].parentid] = parentid

            if np.isnan(parentid) or parentid == "":
                root = True
                vid_dict[skeleton_data.iloc[x].vid] = counter
                # vid_dict[x]=counter

    if parentids[0] == 0:  # to correct for if first node is a branch:
        vid_dict[0] = 0
    edges_dict = dict(sorted(edges_dict.items()))

    parent_nodes_only = skeleton_data[skeleton_data["vid"].isin(parentids)][["x_skeleton", "y_skeleton", "z_skeleton"]].values
    parent_nodes_order = np.array([vid_dict[x] for x in skeleton_data[skeleton_data["vid"].isin(parentids)]["vid"].values])

    parent_nodes_only = parent_nodes_only[parent_nodes_order <= max_node_order]
    parent_nodes_order = parent_nodes_order[parent_nodes_order <= max_node_order]

    # edges =  skeleton_data[skeleton_data['vid'].isin(parentids)][["vid", "parentid"]].values
    ## renumber edges
    remap_dict = {}
    remap_dict[list(edges_dict.items())[0][1]] = 0
    edges_new = []
    counter = 0
    for i, (vid, parentid) in enumerate(edges_dict.copy().items()):
        if vid_dict[vid] <= max_node_order:
            remap_dict[vid] = counter + 1
            counter += 1
        else:
            edges_dict.pop(vid)

            print("intersting")
    # for i, (vid, parentid) in enumerate(edges_dict.items()):
    #     remap_dict[vid]=i+1
    for vid_new, parentid_new in edges_dict.items():
        edges_new.append(np.array([remap_dict[vid_new], remap_dict[parentid_new]]))

    return parent_nodes_only, parent_nodes_order, edges_new


def convert_segmentation2skeleton(df, clustering="dbscan", visualize=False):
    if clustering == "dbscan":
        from sklearn.cluster import DBSCAN

        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.02, min_samples=5, algorithm="auto")
        labels = dbscan.fit_predict(df[["x", "y", "z"]].values)
    elif clustering == "hdbscan":
        import hdbscan

        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
        labels = hdbscan_clusterer.fit_predict(df[["x", "y", "z"]].values)

    else:
        print("clustering method unknown check %s", clustering)

    # Color mapping for clusters
    max_label = labels.max()
    colors = (np.random.rand(max_label, 3) * 255).astype(int)
    colors = np.zeros((len(df), 3))
    for x in range(max_label + 1):
        colors[labels == x] = np.random.rand(3)  # .astype(int)

    df["labels"] = labels
    # df.groupby("labels").mean(["x", "y", "z"]).to_csv("test_pred.csv", index=False)
    S_pred = reconstruct_tree(df.groupby("labels").mean(["x", "y", "z"]).values, df)
    if visualize:
        # for debugging
        # Convert to Open3D Point Cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Visualize the result
        o3d.visualization.draw_geometries([point_cloud], window_name="DBSCAN Clustering")
    return S_pred


def evaluate_skeleton(S_gt, S_pred, method="1", visualize=False):
    th = 0.02  # cm

    ## evaluation using linear sum assignment
    if method == "1":
        cost_matrix = distance_matrix(S_gt.XYZ, S_pred.XYZ)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # , maximize=th) # improved version of hungarion algorithm

        TP = (cost_matrix[row_ind, col_ind] <= th).sum()
        FP = (cost_matrix[row_ind, col_ind] >= th).sum()
        FN = len(S_gt.XYZ) - TP

        ## create dataframe for visualisation.
        df_result = pd.DataFrame(S_gt.XYZ, columns=["x", "y", "z"])
        df_result[["blue", "green", "red"]] = [0, 0, 0]
        dummy = np.full(cost_matrix.shape[0], False)
        dummy[row_ind[cost_matrix[row_ind, col_ind] <= th]] = True
        df_result.loc[dummy, ["blue", "green", "red"]] = [0, 255, 0]  # TP
        df_result.loc[~dummy, ["blue", "green", "red"]] = [0, 0, 255]  # FN
        ##
        df_result_pred = pd.DataFrame(S_pred.XYZ, columns=["x", "y", "z"])
        df_result_pred[["blue", "green", "red"]] = [0, 0, 0]
        dummy = np.full(cost_matrix.shape[1], False)
        dummy[col_ind[cost_matrix[row_ind, col_ind] < th]] = True
        df_result_pred.loc[dummy, ["blue", "green", "red"]] = [0, 200, 0]  # TP
        df_result_pred.loc[~dummy, ["blue", "green", "red"]] = [255, 0, 0]  # FP
        df_result = pd.concat([df_result, df_result_pred])
        print("x")
    else:
        ## evaluation, might be suboptimal because of for loop
        df_result = []
        df_pred_copy = S_pred.XYZ.copy()

        TP = 0
        FP = 0
        FN = 0

        # TODO improve the fact that it is possible that a point is assigned to a suboptimal points as there can be two nodes within 2cm
        # TODO add evaluation of edges
        for query_point in S_gt.XYZ:
            dist = np.linalg.norm(df_pred_copy - query_point, axis=1)
            dist_order = np.argsort(dist)
            # [k, idx, _] = pcd_tree.search_radius_vector_3d(query_point, radius)
            if dist[dist_order[0]] <= th:
                TP += 1

                df_result.append(query_point.tolist() + [0, 255, 0])  # BGR
                df_result.append(df_pred_copy[dist_order[0]].tolist() + [0, 200, 0])  # BGR
                df_pred_copy = df_pred_copy[dist_order[1:]]
            else:
                FN += 1
                df_result.append(query_point.tolist() + [0, 0, 255])  # BGR
        for x in df_pred_copy:
            df_result.append(x.tolist() + [255, 0, 0])  # BGR

        FP = len(df_pred_copy)
        # visualisation of errors
    print("TP", TP)
    print("FP", FP)
    print("FN", FN)

    if visualize:
        # Visualize the result
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(df_result[["x", "y", "z"]].values)
        point_cloud.colors = o3d.utility.Vector3dVector(df_result[["blue", "green", "red"]].values)

        o3d.visualization.draw_geometries([point_cloud], window_name="TP green, FN Red, FP in blue")
        # pd.DataFrame(df_result, columns=["x", "y", "z", "blue", "green", "red"]).to_csv("df_result.csv", index=False)
    print("Finished")


def reconstruct_tree(points, df):
    # Step 1: Calculate the pairwise distances between all points
    distance_matrix = squareform(pdist(points))

    # Step 2: Compute the Minimum Spanning Tree (MST)
    mst_matrix = minimum_spanning_tree(distance_matrix)

    # Step 3: Convert the MST to a list of edges
    mst_edges = np.transpose(mst_matrix.nonzero())
    mst_edges = [np.array([i, j]) for i, j in mst_edges]

    # Print the edges of the MST
    print("Edges in the Minimum Spanning Tree:")
    for edge in mst_edges:
        print(f"Edge between node {edge[0]} and node {edge[1]} with distance {distance_matrix[edge[0], edge[1]]:.2f}")

    S = skeleton.Skeleton(points, mst_edges)
    return S
    fh = plt.figure()
    ax = fh.add_subplot(111, projection="3d")

    vis.plot_skeleton(ax, S)
    vis.plot_pointcloud(ax, df[["x", "y", "z"]].values)
    plt.show()
    return S


def create_skeleton_gt_data(df_gt):
    ## TODO fix node order
    skeleton_data = df_gt.loc[
        ~df_gt["x_skeleton"].isna(), ["x_skeleton", "y_skeleton", "z_skeleton", "vid", "parentid", "edgetype"]
    ]
    parent_nodes_only, parent_node_order, edges = get_only_parent_nodes(skeleton_data)
    S_gt = skeleton.Skeleton(parent_nodes_only, edges)
    return S_gt


if __name__ == "__main__":
    folder = Path("3DTomatoDataset") / "20240607_summerschool_csv" / "annotations"
    plant_id = "Harvest_01_PotNr_179"
    input_file = folder / plant_id / (plant_id + ".csv")
    df_gt = pd.read_csv(input_file)
    S_gt = create_skeleton_gt_data(df_gt)

    folder = Path("./Resources/")
    input_file = folder / (input_file.stem + ".txt")
    df = pd.read_csv(input_file)
    df = df.loc[df["class_pred"] == 4, ["x", "y", "z"]]

    S_pred = convert_segmentation2skeleton(df, "dbscan")

    evaluate_skeleton(S_gt, S_pred, method="1", visualize=True)
    # exit()

    # Perform matching
    # params = {'weight_e':0.01, 'match_ends_to_ends': False,  'use_labels' : False, 'label_penalty' : 1, 'debug': False}
    # corres = skm.skeleton_matching(S_pred, S_gt, params)
    # print("Estimated correspondences: \n", corres)

    # visualize results
    fh = plt.figure()
    ax = fh.add_subplot(111, projection="3d")
    vis.plot_skeleton(ax, S_gt, "b", label="GT")
    vis.plot_skeleton(ax, S_pred, "r", label="Pred")
    # vis.plot_skeleton_correspondences(ax, S_gt, S_pred, corres)
    # vis.plot_skeleton_correspondences(ax, S_pred, S_gt, corres)

    # plt.title("Estimated correspondences between skeletons")
    plt.legend()
    plt.show()
