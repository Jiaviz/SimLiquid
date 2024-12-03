import blenderproc as bproc
import bpy 
from object_print3d_utils import mesh_helpers

import json
import re
import argparse, os, sys, yaml, contextlib
import numpy as np
from time import perf_counter
from tqdm import trange


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str, help="path to config file")
    parser.add_argument("--environment-type", "-et", type=str,
                        help="type of scene, hdri or room")
    parser.add_argument("--num-scenes", "-ns", type=int,
                        help="number of scene to sample")
    parser.add_argument("--cup-poses-num", "-pn", type=int,
                        help="number of pose in every scene")
    parser.add_argument("--output-dir", "-o",
                        type=str, help="path of render result")

    return parser


# add yaml params to command line params
def yaml_to_params(parser):
    args = parser.parse_args()
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    parser = get_parser()
    parser.set_defaults(**cfg_params)
    return parser.parse_args()


@contextlib.contextmanager
def hide_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

@contextlib.contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


class liquid_render:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.render_init(self.cfg)

        self.textures = bproc.loader.load_haven_mat(self.cfg.hdri_path)
        self.plane_desk = bproc.object.create_primitive('PLANE', scale=[2, 2, 1])

        if self.cfg.distractor_objs_num > 0:
            with catchtime() as t:
                print("Loading distactor ...")
                with hide_stdout():
                    self.distractors = self.load_distractor_objs()
            print(f"Done! Execution time: {t():.4f} secs")
    
    def render_init(self, cfg):
        bproc.init()

        # bproc.renderer.set_denoiser("OPTIX")
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"
        
        bproc.renderer.set_light_bounces(diffuse_bounces=5, glossy_bounces=20, max_bounces=45,
                                         transmission_bounces=10, transparent_max_bounces=10, volume_bounces=10)

        bproc.camera.set_intrinsics_from_K_matrix(
            np.array(self.cfg.k_matrix).astype(np.float32), *self.cfg.image_size)

        # activate depth rendering
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

        # activate normal rendering
        bproc.renderer.enable_normals_output()

        bproc.renderer.set_max_amount_of_samples(cfg.render_sample_rate)

    def load_distractor_objs(self):
        bop_parent_path = self.cfg.bop_path
        distractor_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(bop_parent_path, 'ycbv'),  mm2m=True)
        distractor_objs += bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(
            bop_parent_path, 'tless'), model_type='cad', mm2m=True)
        distractor_objs += bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(
           bop_parent_path, 'hb'), mm2m=True)
        distractor_objs += bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_parent_path, 'itodd'), mm2m = True)
        for obj in distractor_objs:
            obj.del_cp("category_id")
            obj.hide(True)

        return distractor_objs
    
    
    def create_environment(self):
        haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(
            self.cfg.hdri_path)
        bproc.world.set_world_background_hdr_img(haven_hdri_path, np.random.uniform(0.7,1))

    # create room with various textures
    def create_room(self, scale=(4, 4, 1)):
        room_planes = [bproc.object.create_primitive('PLANE', scale=scale),
                       bproc.object.create_primitive('PLANE', scale=scale, location=[0, -4, 4],
                                                     rotation=[-1.570796, 0, 0]),
                       bproc.object.create_primitive('PLANE', scale=scale, location=[0, 4, 4],
                                                     rotation=[1.570796, 0, 0]),
                       bproc.object.create_primitive('PLANE', scale=scale, location=[4, 0, 4],
                                                     rotation=[0, -1.570796, 0]),
                       bproc.object.create_primitive('PLANE', scale=scale, location=[-4, 0, 4],
                                                     rotation=[0, 1.570796, 0])]
        for plane in room_planes:
            plane.set_shading_mode("AUTO")
            plane.enable_rigidbody(
                False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

            random_cc_texture = np.random.choice(self.cc_texture)
            plane.replace_materials(random_cc_texture)    

   
    def load_cup_with_liquid(self):
        objs = bproc.loader.load_blend(self.cfg.cups_path)
        for obj in objs:
            # obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
            obj.hide(True)
        
        cups = bproc.filter.by_attr(objs, "name", r'^cup\d+', regex=True)

        cup_collections = {}

        for cup in cups:
            cup.set_cp("bop_dataset_name", "my_own_dataset_name")
            cup.set_cp("category_id", self.cfg.id_dict[cup.get_attr("name")])
            cup_name = cup.get_name()
            liquid = bproc.filter.one_by_attr(objs, "name", cup_name + "_liquid")
            plane = bproc.filter.one_by_attr(objs, "name", cup_name + "_plane")
            appendixes = bproc.filter.by_attr(objs, "name", cup_name + r"_appendix\d+", regex=True)
            cup_collections[cup] = {"liquid":liquid, "plane": plane, "appendixes": appendixes}

        return cup_collections
    
    def get_liquids_material(self):
        materials = bproc.material.collect_all()
        liquid_materials = bproc.filter.by_attr(materials, "name", r'^liquid_\w+\d+', regex=True)
        
        return liquid_materials

    def liquid_sample(self, cup, liquid, plane):
        plane_location = plane.get_location()
        plane_location[2] = np.random.uniform(*self.cfg.liquid_range[cup.get_name()])
        plane.set_location(plane_location)

        bm = mesh_helpers.bmesh_copy_from_object(
            liquid.blender_obj, apply_modifiers=True)
        volume = bm.calc_volume()
        volume = round(volume * 1000000, 1)
        bm.free()

        return volume

    
    def sample_cups_on_desk(self, objs, ground, sample_func, distance_range):
        placed_objs = bproc.object.sample_poses_on_surface(objects_to_sample=objs,
                                                           max_tries=200,
                                                           surface=ground,
                                                           sample_pose_func=sample_func,
                                                           min_distance=distance_range[0],
                                                           max_distance=distance_range[1]) 
        return placed_objs


    def sample_camera_pose_desk(self, objs, cam_poses_num, shell_params):
        cam_poses = 0
        while cam_poses < cam_poses_num:
            poi = bproc.object.compute_poi(objs)
            location = bproc.sampler.shell(
                center=poi, **shell_params)
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                     inplane_rot=np.random.uniform(-0.3, 0.3))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(
                location, rotation_matrix)

            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.5}, bop_bvh_tree, sqrt_number_of_rays=20):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1

        
    def run(self):
        num_scenes = self.cfg.num_scenes
        environment_type = self.cfg.environment_type
        cup_poses_num = self.cfg.cup_poses_num
        output_dir = self.cfg.output_dir
        cup_collections = self.load_cup_with_liquid()
        liquid_materials = self.get_liquids_material()

        for i in trange(num_scenes):
            if environment_type == "hdri":
                self.create_environment()
            elif environment_type == "room":
                self.create_room()
                self.light_sampling()
            else:
                raise NotImplementedError("no environment implemented!")

            sub_dir = os.path.join(output_dir, str(i))
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)

            desk = self.plane_desk 
            desk.replace_materials(np.random.choice(self.textures))
            desk.hide(False)

            sampled_cups = list(np.random.choice(list(cup_collections.keys()), 4, replace=False))
            cup_volume = {}
            sampled_liquids = []

            for cup in sampled_cups:
                liquid_material = np.random.choice(liquid_materials, size=1)[0]
                liquid_name = liquid_material.get_name()
                liquid_label = re.match(r'liquid_(\w+)\d+', liquid_name).group(1)
                
                liquid = cup_collections[cup]["liquid"]
                plane = cup_collections[cup]["plane"]
                
                liquid.set_material(0, liquid_material)
                plane.set_material(0, liquid_material)
                
                volume = self.liquid_sample(cup, liquid, plane)
                cup_volume[cup.get_cp("category_id")] = {"volume": volume, "liquid_label": liquid_label}
                
                cup.hide(False)
                liquid.hide(False)
                for appendix in cup_collections[cup]["appendixes"]:
                    appendix.hide(False)
                
                sampled_liquids.append(liquid)
            
            with open(os.path.join(sub_dir, "volumes.txt"), 'w') as f:
                f.write(json.dumps(cup_volume))
            
            if self.cfg.distractor_objs_num > 0:
                sampled_distractor_objs = list(np.random.choice(self.distractors, size=np.random.randint(1, 10)))
                for obj in sampled_distractor_objs:
                    obj.hide(False)
                placed_objs = sampled_distractor_objs + sampled_cups
            else:
                placed_objs = sampled_cups
            
            def sample_initial_pose(obj):
                obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=desk,
                                                min_height=1, max_height=4, face_sample_range=[0.2, 0.8]))
                obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))
             
            try:
                placed_objs = self.sample_cups_on_desk(
                    objs=placed_objs, ground=desk, sample_func=sample_initial_pose, distance_range=self.cfg.distance_range)
            except ReferenceError:
                print("desk sample failed, skip") 
            
            self.sample_camera_pose_desk(placed_objs, cup_poses_num, self.cfg.shell_params)
            
            for obj in placed_objs:
                if re.match(r"cup\d+", obj.get_name()) and obj.is_hidden():
                    cup_collections[obj]["liquid"].hide(True)
                    for appendix in cup_collections[obj]["appendixes"]:
                        appendix.hide(True)
            
            self.render(bop_output_dir=sub_dir, coco_output_dir=sub_dir, target_objs=sampled_cups)

            for obj in placed_objs:
                obj.hide(True)
            desk.hide(True)
            
            for cup in sampled_cups:
                cup_collections[cup]["liquid"].set_cp("category_id", cup.get_cp("category_id"))
                for appendix in cup_collections[cup]["appendixes"]:
                    appendix.hide(True)
            
            sub_liquid_dir = os.path.join(sub_dir, "liquid")
            self.render(sub_liquid_dir, sub_liquid_dir, sampled_liquids)

            for cup in sampled_cups:
                cup_collections[cup]["liquid"].hide(True)
            
            if environment_type == "room":
                self.light_plane.delete()
                self.light_point.delete()
            

    
    def render(self, bop_output_dir, coco_output_dir, target_objs):

        # render the whole pipeline
        data = bproc.renderer.render()
        seg_data = bproc.renderer.render_segmap(
            map_by=["instance", "class", "name"])
        # Write data in bop format
        if bop_output_dir != "None":
            bproc.writer.write_bop(os.path.join(bop_output_dir, 'bop_data'),
                                   append_to_existing_output=True,
                                   target_objects=target_objs,
                                   depths=data["depth"],
                                   colors=data["colors"],
                                   color_file_format="JPEG",
                                   calc_mask_info_coco=False,
                                   ignore_dist_thres=10)

        # Write data to coco file
        if coco_output_dir != "None":
            bproc.writer.write_coco_annotations(os.path.join(coco_output_dir, 'coco_data'), append_to_existing_output=True,
                                                instance_segmaps=seg_data["instance_segmaps"],
                                                instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                colors=data["colors"],
                                                color_file_format="JPEG")


if __name__ == "__main__":
    args = yaml_to_params(get_parser())
    render = liquid_render(args)
    render.run()
