import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def l2_loss(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angular_distance(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    # Clamp the dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot_product))

def compare_columns(mat1, mat2):
    # Compare each column of the rotation matrices
    num_columns = mat1.shape[1]
    angular_distances = []
    for i in range(num_columns):
        col1 = mat1[:, i]
        col2 = mat2[:, i]
        angular_dist = angular_distance(col1, col2)
        angular_distances.append(angular_dist)
    return np.mean(angular_distances)

def quaternion_to_euler(quaternion):
    r = R.from_quat(quaternion)
    return r #r.as_euler('xyz', degrees=True)

def compute_transformation(reference_quat, target_quat):
    # Compute rotation difference
    reference_rotation = R.from_quat(reference_quat)
    target_rotation = R.from_quat(target_quat)
    relative_rotation = target_rotation * reference_rotation.inv()
    relative_rotation_matrix = relative_rotation.as_matrix()

    return relative_rotation_matrix

def main(colmap_file, vggsfm_file):
    colmap_data = load_json(colmap_file)
    vggsfm_data = load_json(vggsfm_file)

    total_quaternion_loss = 0
    total_position_loss = 0
    common_image_count = 0

    transformation_computed = False
    relative_rotation_matrix = None
    relative_translation = None

    for image_name in tqdm(colmap_data, desc='Compare VGGSfM and Colmap'):
        if image_name in vggsfm_data:
            colmap_quaternion, colmap_position = colmap_data[image_name]
            vggsfm_quaternion, vggsfm_position = vggsfm_data[image_name]

            if not transformation_computed:
                relative_rotation_matrix = compute_transformation(
                    colmap_quaternion, vggsfm_quaternion,
                )
                # relative_translation = relative_rotation_matrix @ np.array(colmap_position) - vggsfm_position
                transformation_computed = True
                continue

            colmap_euler = quaternion_to_euler(colmap_quaternion).as_matrix()
            vggsfm_euler = quaternion_to_euler(vggsfm_quaternion).as_matrix()

            transformed_colmap_rotation_matrix = relative_rotation_matrix @ colmap_euler

            quaternion_loss = compare_columns(transformed_colmap_rotation_matrix, vggsfm_euler)
            position_loss = l2_loss(colmap_euler.T @ colmap_position, vggsfm_euler.T @ vggsfm_position)

            # print(position_loss)

            total_quaternion_loss += quaternion_loss
            total_position_loss += position_loss
            common_image_count += 1

    if common_image_count == 0:
        print("No common images found between the two JSON files.")
        return

    average_quaternion_loss = total_quaternion_loss / common_image_count
    average_position_loss = total_position_loss / common_image_count

    print(f"Average angular distance for Euler angle: {average_quaternion_loss}")
    print(f"Average L2 loss for positions: {average_position_loss}")

colmap_file = 'mast3r_cams.json'
vggsfm_file = 'vggsfm_cams.json'
main(colmap_file, vggsfm_file)
