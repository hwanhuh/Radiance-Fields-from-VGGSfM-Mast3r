import random
import time
import argparse
from pathlib import Path
import imageio.v3 as iio
import numpy as onp
import tyro
import json
import trimesh
import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from viser.extras.colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3d_binary,
)

def main(
    images_path: str,
    colmap_path: str, 
    downsample_factor: int = 4,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    images_path = Path(images_path)
    colmap_path = Path(colmap_path)
    
    server = viser.ViserServer()
    server.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    try:
        cameras = read_cameras_binary(colmap_path / "cameras.bin")
        images = read_images_binary(colmap_path / "images.bin")
    except:
        cameras = read_cameras_text(colmap_path / "cameras.txt")
        images = read_images_text(colmap_path / "images.txt")
    try:
        points3d = read_points3d_binary(colmap_path / "points3D.bin")
    except:
        mesh = trimesh.load(colmap_path / "points3D.ply")
        points3d = mesh.vertices 
        colors3d = mesh.colors
    
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.add_gui_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.add_gui_number("Point size", initial_value=0.0003)
    RTs = {}

    # Interpret the images and cameras.
    img_ids = [im.id for im in images.values()]
    random.shuffle(img_ids)
    img_ids = sorted(img_ids[: gui_frames.value])

    def set_image_frustums() -> None:
        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids, desc='Loading Images'):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / img.name
            if not image_filename.exists():
                continue

            RTs[f'{img.name}'] = [img.qvec.tolist(), img.tvec.tolist()]

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            frame = server.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )

            # if cam.model != "PINHOLE":
            #     print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            frustum = server.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * onp.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.05,
                image=image,
            )
            attach_callback(frustum, frame)

        # with open('cams.json', 'w') as json_file:
        #     json.dump(RTs, json_file, indent=4)

    def visualize_colmap(load_images) -> None:
        if load_images:
            set_image_frustums()
        """Send all COLMAP elements to viser for visualization"""
        try:
            points = onp.array([points3d[p_id].xyz for p_id in points3d])
            colors = onp.array([points3d[p_id].rgb for p_id in points3d])
            points_selection = onp.random.choice(
                points.shape[0], gui_points.value, replace=False
            )
            points = points[points_selection]
            colors = colors[points_selection]

            server.add_point_cloud(
                name="/colmap/pcd",
                points=points,
                colors=colors,
                point_size=gui_point_size.value,
            )
        except:
            server.add_point_cloud(
                name="/colmap/pcd",
                points=points3d,
                colors=onp.array(colors3d)[:, :3],
                point_size=gui_point_size.value,
            )
    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    server.reset_scene()
    visualize_colmap(True)

    while True:
        if need_update:
            need_update = False
            visualize_colmap(False)

        time.sleep(1e-3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--colmap_path', type=str, required=True, help='Path to the colmap model including vggsfm output and mast3r')
    parser.add_argument('--downsample', type=int, default=4)

    args = parser.parse_args()
    main(args.image_dir, args.colmap_path, args.downsample)