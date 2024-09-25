import time
from tqdm import tqdm
import numpy as np
import pinocchio as pin
from pathlib import Path
import json
import hppfcl
from typing import Union

from src.scene import DiffColScene
from src.optimizer import optim, three_phase_optim
from config import (MESHES_PATH,
                    MESHES_DECOMP_PATH,
                    FLOOR_MESH_PATH,
                    DATASETS_PATH,
                    POSES_OUTPUT_PATH,
                    FLOOR_POSES_PATH)

from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam, load_meshes, load_meshes_decomp, load_csv, load_mesh

def create_decomposed_mesh(mesh_loader, dir_path: str):
    meshes = []
    for path in Path(dir_path).iterdir():
        if path.suffix == ".ply" or path.suffix == ".obj":
            mesh = mesh_loader.load(str(path), scale=np.array(3*[0.001]))
            mesh.buildConvexHull(True, "Qt")
            meshes.append(mesh.convex)
    return meshes

def save_optimized_bproc(ds_name, params, visualize=False):
    """
    Optimizes poses of objects in BOP dataset using collision detection.
    Inputs:
    - ds_name: str, name of the dataset
    - params: dict, optimization parameters
    - visualize: bool, whether to visualize the optimization process
    Returns: None
    """
    col_req = hppfcl.DistanceRequest()

    print("Loading decomposed meshes...")
    path_objs_decomposed = MESHES_DECOMP_PATH / ds_name
    mesh_objs_dict_decomposed = load_meshes_decomp(path_objs_decomposed)
    print("Loading meshes...")
    path_objs_convex = MESHES_PATH / ds_name
    mesh_objs_dict = load_meshes(path_objs_convex)
    path_stat_objs = [FLOOR_MESH_PATH]
    static_meshes = [load_mesh(str(p)) for p in path_stat_objs]
    
    dataset_path = DATASETS_PATH / ds_name
    path_wMo_all = dataset_path / "happypose/outputs"
    gt_cam_path = dataset_path / "train_pbr/000000/scene_camera.json"
    wMo_lst_all = []
    wMs_lst = []
    label_objs_all = []
    scene_idxs = []
    with open(gt_cam_path, "r") as f:
        gt_cam = json.load(f)
    for wMo_path in path_wMo_all.iterdir():
        scene_idx = int(wMo_path.name.split("_")[-1].split(".")[0])
        with open(wMo_path, "r") as f:
            wMo_json = json.load(f)
        wMo_lst = []
        label_objs = []
        se3_cam = get_se3_from_bp_cam(gt_cam[str(scene_idx)])
        for i in range(len(wMo_json)):
            label = int(wMo_json[i]["label"])
            wMo = get_se3_from_mp_json(wMo_json[i])
            wMo_lst.append(wMo)
            label_objs.append(label)
        wMo_lst_all.append(wMo_lst)
        wMs_lst.append(se3_cam)
        label_objs_all.append(label_objs)
        scene_idxs.append(scene_idx)

    save_dir_path = dataset_path / "happypose/outputs_coll"
    for i in tqdm(range(len(wMo_lst_all))):
        curr_meshes = []
        curr_meshes_decomp = []
        for l in label_objs_all[i]:
            curr_meshes.append(mesh_objs_dict[l])
            curr_meshes_decomp.append(mesh_objs_dict_decomposed[l])
        wMo_lst = wMo_lst_all[i]
        dc_scene = DiffColScene(curr_meshes, static_meshes, [wMs_lst[i]], curr_meshes_decomp, pre_loaded_meshes=True)
        X = optim(dc_scene, wMo_lst, col_req, params, visualize)
        to_json = []
        for j in range(len(X)):
            xyzquat = pin.SE3ToXYZQUAT(X[j])
            json_dict = {"label":str(label_objs_all[i][j]), "TWO":[list(xyzquat[3:]), list(xyzquat[:3])]}
            to_json.append(json_dict)
        save_data_path = save_dir_path / f"object_data_{scene_idxs[i]}.json"
        save_data_path.write_text(json.dumps(to_json))


def save_optimized_floor(dataset_name:str, floor_name:str, params:dict = None, vis:bool = None):
    """Optimizes pose of floor in YCBV BOP dataset using collision detection.
    Inputs:
    - dataset_name: str, name of the dataset
    - floor_name: str, name of the floor
    - params: dict, optimization parameters
    - vis: bool, whether to visualize the optimization process
    Returns: None
    """

    rigid_objects = load_meshes(MESHES_PATH / dataset_name)
    rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / dataset_name)
    floor_mesh, floor_se3s = load_static(floor_name)
    col_req = hppfcl.DistanceRequest()
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / dataset_name, convex=False)
        curr_meshes_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]
    else:
        curr_meshes_vis = None

    optimized_floor = {}

    for scene in tqdm(floor_se3s):
        with open(DATASETS_PATH / dataset_name / f"{int(scene):06d}" / "scene_gt.json", "r") as f:
            gt_poses_all = json.load(f)
        optimized_floor[scene] = {}
        for im in tqdm(floor_se3s[scene]):
            curr_stat_meshes = []
            curr_stat_meshes_decomp = []
            wMs_lst = []
            if vis:
                curr_meshes_stat_vis = []
            else:
                curr_meshes_stat_vis = None
            # Load info about each object in the scene
            for obj in gt_poses_all[im]:
                wMs_lst.append(pin.SE3(np.array(obj["cam_R_m2c"]).reshape(3,3), np.array(obj["cam_t_m2c"])/1000))
                curr_stat_meshes.append(rigid_objects[str(obj["obj_id"])])
                curr_stat_meshes_decomp.append(rigid_objects_decomp[str(obj["obj_id"])])
                if vis:
                    curr_meshes_stat_vis.append(rigid_objects_vis[str(obj["obj_id"])])
            wMo = floor_se3s[str(scene)][str(im)]
            wMo_lst, curr_meshes = (None, None) if wMo is None else ([pin.SE3(np.array(wMo["R"]), np.array(wMo["t"]))], [floor_mesh])
            if wMo_lst is None:
                optimized_floor[scene][im] = None
                continue
            dc_scene = DiffColScene(curr_meshes, curr_stat_meshes, wMs_lst, [], curr_stat_meshes_decomp, pre_loaded_meshes=True)
            X = optim(dc_scene, wMo_lst, col_req, params, curr_meshes_vis, curr_meshes_stat_vis)
            optimized_floor[scene][im] = {"R": X[0].rotation.tolist(), "t": (X[0].translation).tolist()}
        with open(FLOOR_POSES_PATH / (floor_name[:-5] + "_optimized.json"), "w") as f:
            json.dump(optimized_floor, f)

def load_static(floor_poses_name:str):
    """Loads floor mesh and poses from JSON file."""
    mesh_loader = hppfcl.MeshLoader()
    mesh = mesh_loader.load(str(FLOOR_MESH_PATH), np.array(3*[0.01]))
    mesh.buildConvexHull(True, "Qt")
    floor_mesh = mesh.convex
    with open(FLOOR_POSES_PATH / floor_poses_name, "r") as f:
        floor_se3s = json.load(f)
    return floor_mesh, floor_se3s

def save_optimized_bop(input_csv_name:str, output_csv_name:str,
                       dataset_name: str, use_floor:Union[None, str],
                       params:dict = None, vis:bool = False):
    """
    Optimizes poses of objects in BOP dataset using collision detection.
    Inputs:
    - input_csv_name: str, name of the input CSV file
    - output_csv_name: str, name of the output CSV file
    - dataset_name: str, name of the dataset
    - use_floor: str, name of the floor
    - params: dict, optimization parameters
    - vis: bool, whether to visualize the optimization process
    Returns: None
    """
    meshes_ds_name = ""
    if dataset_name[:4] == "ycbv":
        meshes_ds_name = "ycbv"
    elif dataset_name[:5] == "tless":
        meshes_ds_name = "tless"
    else:
        meshes_ds_name = dataset_name
    rigid_objects = load_meshes(MESHES_PATH / meshes_ds_name)
    rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / meshes_ds_name)
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / meshes_ds_name, convex=False)
    scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / input_csv_name)
    scenes_gt = load_csv(POSES_OUTPUT_PATH / dataset_name / f"gt_{dataset_name}-test.csv")     ############################################## DELETE
    if use_floor != None:
        floor_mesh, floor_se3s = load_static(use_floor)
    if vis and use_floor != None:
        floor_mesh_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]
    else:
        floor_mesh_vis = None
    col_req = hppfcl.DistanceRequest()

    with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")

    ############################################## DELETE
    ### RMC
    # from robomeshcat import Scene, Object
    # scene_rmc = Scene()
    # if use_floor:
    #     floor_rmc = Object.create_mesh(
    #         FLOOR_MESH_PATH.parent / "floor" / "floor.obj",
    #         scale=0.001,
    #         texture=FLOOR_MESH_PATH.parent / "floor" / "Wood001_2K-JPG_Color.jpg",
    #         color=[0.5] * 3,
    #         )
    #     scene_rmc.add_object(floor_rmc)
    # objects_all_rmc = {}
    # print("Loading RMC meshes")
    # for mesh_dir_path in tqdm((MESHES_PATH / meshes_ds_name).iterdir()):
    #     l = int(mesh_dir_path.name)
    #     mesh_path = mesh_dir_path /f"obj_{l:06d}.ply"
    #     texture_path = mesh_dir_path /f"obj_{l:06d}.png"
    #     o = Object.create_mesh(
    #         mesh_path,
    #         scale=0.001,
    #         texture=texture_path,
    #         color=[1] * 3,
    #         )
    #     objects_all_rmc[str(l)] = o
    ####
    ############################################## DELETE

    for scene in tqdm(scenes):
        for im in tqdm(scenes[scene]):
            curr_labels = []
            curr_meshes = []
            curr_meshes_decomp = []
            if vis:
                curr_meshes_vis = []
            else:
                curr_meshes_vis = None
            wMo_lst = []
            # Load info about each object in the scene
            for label, R_o, t_o in zip(scenes[scene][im]["obj_id"], scenes[scene][im]["R"], scenes[scene][im]["t"]):
                R_o = np.array(R_o).reshape(3, 3)
                t_o = np.array(t_o)
                wMo = pin.SE3(R_o, t_o)
                curr_labels.append(label)
                curr_meshes.append(rigid_objects[label])
                if vis:
                    curr_meshes_vis.append(rigid_objects_vis[label])
                curr_meshes_decomp.append(rigid_objects_decomp[label])
                wMo_lst.append(wMo)
            if use_floor:
                wMs = floor_se3s[str(scene)][str(im)]
                wMs, stat_meshes = ([], []) if wMs is None else ([pin.SE3(np.array(wMs["R"]), np.array(wMs["t"]))], [floor_mesh])
                dc_scene = DiffColScene(curr_meshes, stat_meshes, wMs, curr_meshes_decomp, pre_loaded_meshes=True)
            else:
                dc_scene = DiffColScene(curr_meshes, [], [], curr_meshes_decomp, pre_loaded_meshes=True)
            start_time = time.time()
            if params["g_grad_scale"] == 0:
                X = optim(dc_scene, wMo_lst, col_req, params, curr_meshes_vis, floor_mesh_vis)
            else:
                X = three_phase_optim(dc_scene, wMo_lst, col_req, params, curr_meshes_vis, floor_mesh_vis)
            optim_time = (time.time() - start_time)


            ############################################## DELETE
            ### RMC
            # floor_rmc.pose = np.eye(4) #wMs[0].homogeneous
            # curr_objects_rmc = [objects_all_rmc[l] for l in curr_labels]
            # for object_rmc in list(scene_rmc.objects.values()):
            #     if object_rmc not in curr_objects_rmc and object_rmc != floor_rmc:
            #         scene_rmc.remove_object(object_rmc)
            # for curr_object_rmc in curr_objects_rmc:
            #     if curr_object_rmc not in list(scene_rmc.objects.values()):
            #         scene_rmc.add_object(curr_object_rmc)
            # ### ANIMATION
            # for X_init, obj_curr in zip(X_lst[0], curr_objects_rmc):
            #     obj_curr.pose = (wMs[0].inverse()*X_init).homogeneous
            # with scene_rmc.video_recording(f"{dataset_name}_{scene}_{im}.mp4", 15):
            #     for X_lst_curr in X_lst[:len(X_lst)//4]:
            #         for X_curr, obj_curr in zip(X_lst_curr, curr_objects_rmc):
            #             obj_curr.pose = (wMs[0].inverse()*X_curr).homogeneous
            #         scene_rmc.render()
            ### ANIMATION END
            # if len(wMs) == 0:
            #     continue
            # wMo_lst_gt = [] 
            # curr_labels_gt = []
            # for label, R_o, t_o in zip(scenes_gt[scene][im]["obj_id"], scenes_gt[scene][im]["R"], scenes_gt[scene][im]["t"]):
            #     R_o = np.array(R_o).reshape(3, 3)
            #     t_o = np.array(t_o)
            #     wMo = pin.SE3(R_o, t_o)
            #     wMo_lst_gt.append(wMo)
            #     curr_labels_gt.append(label)
            # for label in curr_labels:
            #     assert label in curr_labels_gt
        
            # # Add objects
            # while True:
            #     vis_configuration = input("Visualize configuration (i/o/g/q): ")
            #     time.sleep(3)
            #     if vis_configuration == "i":
            #         for l, T in zip(curr_labels, wMo_lst):
            #             objects_all_rmc[l].pose = (wMs[0].inverse()*T).homogeneous
            #     elif vis_configuration == "o":
            #         for l, T in zip(curr_labels, X):
            #             objects_all_rmc[l].pose = (wMs[0].inverse()*T).homogeneous
            #     elif vis_configuration == "g":
            #         for l, T in zip(curr_labels_gt, wMo_lst_gt):
            #             objects_all_rmc[l].pose = (wMs[0].inverse()*T).homogeneous
            #     elif vis_configuration == "q":
            #         break
            # ####
            ############################################## DELETE
            
            for i in range(len(X)):
                # One CSV row
                R = " ".join(str(item) for item in X[i].rotation.reshape(9).tolist())
                t = " ".join(str(item) for item in (X[i].translation*1000).tolist())
                csv_line = [scene, im, curr_labels[i], 1.0, R, t, optim_time]
                with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "a") as f:
                    f.write(",".join([str(x) for x in csv_line]) + "\n")

if __name__ == "__main__":
    params = {
        "N_step": 1000,
        "g_grad_scale": 1,
        "coll_grad_scale": 1,
        "coll_exp_scale": 0,
        "learning_rate": 0.0001,
        "step_lr_decay": 1,
        "step_lr_freq": 1000,
        "std_xy_z_theta": [0.05, 0.49, 0.26],
        "method": "GD",
        "method_params": None
    }
    
    floor_file_names = {"hopevideo":"hope_bop_floor_poses_1mm_res_optimized.json",
                        "tless":"tless_bop_floor_poses_1mm_res_optimized.json",
                        "ycbv":"ycbv_bop_floor_poses_1mm_res_optimized.json",
                        "ycbvone":"ycbv_one_synt_floor_gt.json",
                        "tlessone":"tless_one_synt_floor_gt.json"}
    floor_names = ["optimized", "none"]

    input_csv_names = {"hopevideo":"refiner-final-filtered_hopevideo-test.csv",
                       "tless":"refiner-final-filtered_tless-test.csv",
                       "ycbv":"refiner-final-filtered_ycbv-test.csv",
                       "ycbvone":"refiner-final-filtered_ycbvone-test.csv",
                       "tlessone":"refiner-final-filtered_tlessone-test.csv"} # INPUT
    
    dataset_names = ["hopevideo","ycbv","tless","ycbvone","tlessone"] # INPUT
    vis = False #INPUT

    #dataset_name = dataset_names[int(sys.argv[1])]
    dataset_name = "ycbv"
    coll_grad_scales = [0.2, 0.5, 1, 2, 5, 10]
    coll_exp_scales = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    #params["coll_grad_scale"] = coll_grad_scales[int(sys.argv[1])]
    #params["coll_exp_scale"] = coll_exp_scales[int(sys.argv[2])]
    #floor_name = floor_names[int(sys.argv[3])]
    floor_name = floor_names[0]

    input_csv_name = input_csv_names[dataset_name]
    if floor_name == "none":
        use_floor = None
        if params["g_grad_scale"] != 0:
            exit()
    else:
        use_floor = floor_file_names[dataset_name]
    

    output_csv_name = (f"central_difference/"
                       f"{params['g_grad_scale']}-"
                       f"{params['coll_grad_scale']}-"
                       f"{params['learning_rate']}-"
                       f"{params['step_lr_decay']}-"
                       f"{params['step_lr_freq']}-"
                       f"{params['std_xy_z_theta'][0]}-{params['std_xy_z_theta'][1]}-{params['std_xy_z_theta'][2]}-"
                       f"{params['coll_exp_scale']}-"
                       f"{floor_name}_"
                       f"{dataset_name}-test").replace(".","") + ".csv"
    print(f"File name: {output_csv_name}")
    save_optimized_bop(input_csv_name, output_csv_name, dataset_name, use_floor, params, vis)