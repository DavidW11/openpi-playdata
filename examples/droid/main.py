# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import csv
from openai import OpenAI
import base64
from PIL import Image
import io
# added for data saving
import cv2
import h5py
import time

faulthandler.enable()


client = OpenAI()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

client = OpenAI(
    # This is the default and can be omitted
    api_key = open("/home/tennyyin/openpi-playdata/examples/droid/openai_apikey.txt", "r").read().strip()
)
@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "31177322"  # e.g., "24259877"
    right_camera_id: str = "38872458"  # e.g., "24514023"
    wrist_camera_id: str = "10501775"  # e.g., "13062452"

    # Policy parameters
    external_camera: str = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 200
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    save_video: bool = False  # whether to save video of rollouts

    long_data_collection: bool = False  # whether to do long data collection sessions (many rollouts without resetting env)

    continue_from_last: bool = False  # whether to continue data collection from last session

    num_trajectories_to_collect: int = 10

    date_str: str = None

    save_base_dir: str = "/home/tennyyin/openpi-playdata/examples/droid/data"


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    max_count = args.num_trajectories_to_collect # number of rollouts per session

    # get string for today's date
    if args.date_str is None:
        date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    else:
        date_str = args.date_str
    
    # check args.save_base_dir to see if there s already a folder for today's date. if so, append a "_2", "_3", etc.
    if not args.long_data_collection:
        data_collection_idx = 1
        save_root_dir = None
        while True:
            date_folder = os.path.join(args.save_base_dir, f"{date_str}_{data_collection_idx}")
            if not os.path.exists(date_folder):
                save_root_dir = date_folder
                if not args.continue_from_last:
                    os.makedirs(date_folder, exist_ok=True)
                break
            data_collection_idx += 1
        
        if args.continue_from_last:
            data_collection_idx -= 1
            save_root_dir = os.path.join(args.save_base_dir, f"{date_str}_{data_collection_idx}")
            print(f"Continuing data collection in existing folder: {save_root_dir}")
    else:
        # long play data is saved just in date_str/
        save_root_dir = os.path.join(args.save_base_dir, f"{date_str}_long")
        os.makedirs(save_root_dir, exist_ok=True)

    while True:
        # =========================================================================
        # ====================== Data Collection Loop Starts ======================
        # =========================================================================

        count = 0
        instruction = ""
        
        if not args.long_data_collection:
            max_count = args.num_trajectories_to_collect # number of rollouts per session
        else:
            max_count = 1000000  # effectively infinite rollouts in long data collection mode
            args.continue_from_last = True  # always continue from last in long data collection mode

        timestamp_folder = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
        
        # TODO is this needed?
        if args.save_video:
            video = []

        if args.continue_from_last:
            # identify last collected trajectory
            existing_traj_folders = [
                d for d in os.listdir(save_root_dir) if os.path.isdir(os.path.join(save_root_dir, d))
            ]
            if existing_traj_folders:
                existing_traj_indices = [
                    int(d) for d in existing_traj_folders if d.isdigit()
                ]
                if existing_traj_indices:
                    last_traj_idx = max(existing_traj_indices)
                    # check if the folder is complete
                    traj_folder = os.path.join(save_root_dir, f"{last_traj_idx}")
                    h5_path = os.path.join(traj_folder, "trajectory.h5")
                    if not os.path.exists(h5_path):
                        print(f"Last trajectory folder {traj_folder} is incomplete, resuming from there.")
                        # remove all files in the folder
                        for f in os.listdir(traj_folder):
                            os.remove(os.path.join(traj_folder, f))
                        count = last_traj_idx
                    else:
                        count = last_traj_idx + 1
                    print(f"Resuming from trajectory index {count}.")

        # =====================================================================================================
        # ==================================== Data Collection Loop Starts ====================================
        # =====================================================================================================

        # Variables to keep track
        last_instruction_scene_reset_count = 0
        last_instruction_detect_oob_count = 0
        relax_safety_filter_count = 0
        enable_safety_filter = True

        critical_failure_occurred = 0

        while count < max_count:

            if critical_failure_occurred >= 10:
                print("Critical failures occurred multiple times, ending data collection session.")
                raise RuntimeError("Too many critical failures during data collection.")
            
            count += 1
            print(f"Starting rollout {count}/{max_count}...")

            left_base_cam = []
            right_base_cam = []
            wrist_cam = []
            joint_position = []
            gripper_position = []
            cartesian_position = []
            action_list = []

            traj_folder = os.path.join(save_root_dir, f"{count}")
            os.makedirs(traj_folder, exist_ok=True)
            h5_path = os.path.join(traj_folder, "trajectory.h5")
            
            # Rollout parameters
            actions_from_chunk_completed = 0
            pred_action_chunk = None
            bar = tqdm.tqdm(range(args.max_timesteps))
            print("Running rollout... press Ctrl+C to stop early.")
            curr_obs = _extract_observation(
                args,
                env.get_observation(),
                save_to_disk=True,
            )

            image_right = curr_obs[f"{args.external_camera}_image"]
            image_wrist = curr_obs["wrist_image"]

            # Get instruction from user
            print("Generating instruction...")
            instruction = generate_prompt(image_right, image_wrist, object="none")
            print(f"Instruction: {instruction}")

            if is_reset_command(instruction):
                last_instruction_detect_oob_count += 1
            else:
                last_instruction_detect_oob_count = 0

            if last_instruction_scene_reset_count >= 3 and last_instruction_detect_oob_count >= 3:
                # if we have reset the scene 3 times trying to reset an object, will relax safety filter
                enable_safety_filter = False
                relax_safety_filter_count += 1

            # reset state to before
            if relax_safety_filter_count >= 5:
                enable_safety_filter = True
                relax_safety_filter_count = 0
                critical_failure_occurred += 1

            for t_step in bar:
                start_time = time.time()
                try:
                    all_obs = env.get_observation()
                    # Get the current observation
                    curr_obs = _extract_observation(
                        args,
                        all_obs,
                        # Save the first observation to disk
                        save_to_disk=t_step == 0,
                    )

                    if args.save_video:
                        video.append(curr_obs[f"{args.external_camera}_image"])

                    # Send websocket request to policy server if it's time to predict a new chunk
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                        actions_from_chunk_completed = 0

                        # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                        # and improve latency.
                        request_data = {
                            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                                curr_obs[f"{args.external_camera}_image"], 224, 224
                            ),
                            "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                            "observation/joint_position": curr_obs["joint_position"],
                            "observation/gripper_position": curr_obs["gripper_position"],
                            "prompt": instruction,
                        }

                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                            pred_action_chunk = policy_client.infer(request_data)["actions"]
                        # assert pred_action_chunk.shape == (15, 8)

                    # Select current action to execute from chunk
                    action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1

                    # Binarize gripper action
                    if action[-1].item() > 0.5:
                        # action[-1] = 1.0
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        # action[-1] = 0.0
                        action = np.concatenate([action[:-1], np.zeros((1,))])

                    # clip all dimensions of action to [-1, 1]
                    action = np.clip(action, -1, 1)

                    # safety check for joint limits
                    all_joint_limits = [
                        [-2.5, 2.5], # 0
                        [-1.5, 1.5], # 1
                        [-2.5, 2.5], # 2
                        [-3.0, 0], # 3 vertical joint
                        [-2.5, 2.5], # 4
                        [0, 3.5], # 5
                        [-2.5, 2.5] # 6
                    ]

                    # Safety Check
                    need_reset = False
                    need_reset_strict = False

                    all_joint_positions = curr_obs["joint_position"]

                    # cartesian position of end-effector in world frame
                    all_cartesian_positions = curr_obs["cartesian_position"]
                    # print(f"Current end-effector position: {all_cartesian_positions[0]:.3f}, {all_cartesian_positions[1]:.3f}, {all_cartesian_positions[2]:.3f}")
                    # print(f"***Current positions: {all_cartesian_positions}")
                    # Compute intersection of end-effector look-at vector with plane z=0
                    x, y, z, roll, pitch, yaw = all_cartesian_positions

                    # Check for gimbal lock (pitch near +/- 90 deg)
                    gimbal_lock_threshold = np.pi/2 - 0.1  # 10 degrees from singularity
                    if abs(pitch) > gimbal_lock_threshold:
                        print(f"WARNING: Approaching gimbal lock! pitch = {np.degrees(pitch):.1f}°")

                    # Compute rotation matrix from RPY (ZYX convention)
                    cr, sr = np.cos(roll), np.sin(roll)
                    cp, sp = np.cos(pitch), np.sin(pitch)
                    cy, sy = np.cos(yaw), np.sin(yaw)

                    # Additional gimbal lock check: when cp ≈ 0, rotation matrix becomes degenerate
                    if abs(cp) < 0.01:  # cos(pitch) ≈ 0 means pitch ≈ ±90°
                        print(f"GIMBAL LOCK DETECTED! pitch = {np.degrees(pitch):.1f}°, cos(pitch) = {cp:.4f}")

                    # Rotation matrix R_ZYX (world to end-effector)
                    R = np.array([
                        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                        [-sp,   cp*sr,            cp*cr]
                    ])

                    # 180° rotation around x-axis transformation from world to end-effector frame
                    R_x_180 = np.array([
                        [1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]
                    ])

                    # Apply the x-axis rotation: R_corrected = R @ R_x_180
                    R_corrected = R @ R_x_180

                    # Look-at direction is the z-axis of the corrected end-effector frame (3rd column)
                    look_at_x = R_corrected[0, 2]
                    look_at_y = R_corrected[1, 2]
                    look_at_z = R_corrected[2, 2]


                    # Find intersection with plane z=0.2: point = [x,y,z] + t*[look_at_x, look_at_y, look_at_z]
                    # At intersection: z + t*look_at_z = 0.2, so t = (0.2 - z)/look_at_z
                    if abs(look_at_z) > 1e-6:  # Avoid division by zero
                        t = (0.2 - z) / look_at_z
                        intersection_x = x + t * look_at_x
                        intersection_y = y + t * look_at_y
                        # print(f"Look-at intersection with z=0.2: ({intersection_x:.3f}, {intersection_y:.3f}, 0.0)")
                        # check if intersection within boundary
                        if not (0.22 <= intersection_x <= 0.8 and -0.2 <= intersection_y <= 0.35):
                            print(f"Look-at intersection outside of boundary ({intersection_x:.3f}, {intersection_y:.3f}), resetting environment...")
                            need_reset = True
                    else:
                        need_reset = True
                        print("Look-at vector is parallel to z=0.2 plane, no intersection")

                    # check eff position limits
                    if not need_reset:
                        eff_x, eff_y, eff_z = all_cartesian_positions[0:3]
                        if not (0 <= eff_x <= 1.0 and -0.6 <= eff_y <= 0.6 and 0.15 <= eff_z <= 0.8):
                            print(f"End-effector position limit exceeded ({eff_x}, {eff_y}, {eff_z}), resetting environment...")
                            need_reset = True
                    
                    # Check joint rotation limits
                    if not need_reset:
                        for j in range(7):
                            if not (all_joint_limits[j][0] <= all_joint_positions[j] <= all_joint_limits[j][1]):
                                print(f"Joint {j+1} limit exceeded {all_joint_positions[j]}, resetting environment...")
                                need_reset_strict = True
                                break

                    if need_reset_strict or (need_reset and not enable_safety_filter):
                        last_instruction_scene_reset_count += 1
                        env.reset()
                        # sleep for 2 seconds to allow environment to reset
                        time.sleep(3)
                        break
                    else:
                        last_instruction_scene_reset_count = 0

                    env.step(action)

                    all_image_obs = all_obs["image"]

                    # reshape images
                    left_external_image = all_image_obs[f"{args.left_camera_id}_left"]
                    left_external_image_resized = cv2.resize(left_external_image, (320, 180))
                    right_external_image = all_image_obs[f"{args.right_camera_id}_left"]
                    right_external_image_resized = cv2.resize(right_external_image, (320, 180))

                    # Convert to RGB
                    left_external_image_resized = left_external_image_resized[..., :3]
                    left_external_image_resized = left_external_image_resized[..., ::-1]
                    right_external_image_resized = right_external_image_resized[..., :3]
                    right_external_image_resized = right_external_image_resized[..., ::-1]

                    wrist_image = curr_obs["wrist_image"]
                    wrist_image_resized = cv2.resize(wrist_image, (320, 180))

                    # append to arrays
                    left_base_cam.append(left_external_image_resized)
                    right_base_cam.append(right_external_image_resized)
                    wrist_cam.append(wrist_image_resized)
                    joint_position.append(curr_obs["joint_position"])
                    gripper_position.append(curr_obs["gripper_position"])
                    cartesian_position.append(curr_obs["cartesian_position"])
                    action_list.append(action)

                    # Sleep to match DROID data collection frequency
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                except KeyboardInterrupt:
                    break
            
            print("Resetting the environment for the next rollout...")
            os.makedirs("/home/tennyyin/openpi-playdata/examples/droid/results", exist_ok=True)

            # Save the data
            # discard the trajectory if length is too short
            trajectory_length = len(left_base_cam)
            if trajectory_length < 50:
                print(f"Trajectory too short (length {trajectory_length}), discarding...")
                count -= 1
            else:
            
                with open("/home/tennyyin/openpi-playdata/examples/droid/results/observations_" + timestamp_folder + ".csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([instruction])

                left_base_cam = np.stack(left_base_cam)
                right_base_cam = np.stack(right_base_cam)
                wrist_cam = np.stack(wrist_cam)
                joint_position = np.stack(joint_position)
                gripper_position = np.stack(gripper_position)
                cartesian_position = np.stack(cartesian_position)
                action_list = np.stack(action_list)

                # save one video per view
                view_keys = ["left_base_cam", "right_base_cam", "wrist_cam"]
                for i, video in enumerate([left_base_cam, right_base_cam, wrist_cam]):
                    save_filename = os.path.join(traj_folder, f'{view_keys[i]}.mp4')
                    image_clip = ImageSequenceClip(list(video), fps=10)
                    image_clip.write_videofile(save_filename, codec="libx264")

                with h5py.File(h5_path, "a") as f:
                    # traj_name = f"trajectory_{count}"
                    traj_name = "data"

                    # Create a new group for this trajectory
                    grp = f.create_group(traj_name)

                    # Create datasets for each array
                    grp.create_dataset("joint_position", data=joint_position)
                    grp.create_dataset("gripper_position", data=gripper_position)
                    grp.create_dataset("cartesian_position", data=cartesian_position)
                    grp.create_dataset("action", data=action_list)

                    # add metadata as attributes
                    grp.attrs["trajectory_index"] = count


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):  
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    # left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    # left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        # "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }

def generate_image(image):
    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8")
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)

    return base64.b64encode(buffered.read()).decode("utf-8")

def is_reset_command(s):
    keywords = ["toward", "center", "table"]
    s_lower = s.lower()
    return all(word in s_lower for word in keywords)

def generate_prompt(image_right, image_wrist, object):
    base64_image_right = generate_image(image_right)
    base64_image_wrist = generate_image(image_wrist)

    user_input = """
    You are a robot that is trying to randomly manipulate/arrange objects on the table in a square region marked by tape.
    First, observe if any object is outside of the square workspace area. If so, please output: "move the <> towards the center of the table".
    Otherwise, please output a task to perform. Possible instructions might include:
    1) Put the <> in the <>.
    2) Pick up the <> and put it near the <>.
    3) Place the <> next to/on the <left/right/front/back> of the <>.
    4) Remove the <> from the <> and put it on the table.
    5) Clean up the table.
    Choose random objects from the existing ones and fill in the <> accordingly. Try to equally interact with all objects on the table. Feel free to modify the prompt by a little bit.
    Only output the concise task, and do not include any other explanations.
    Use color to help the robot better identify the objects, e.g. ("brown cup").
    """

    response = client.responses.create(
        model = "gpt-5-mini",
        input = [
            {"role": "user", "content": [
                {"type": "input_text", "text": user_input},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image_right}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image_wrist}"}
            ]}
        ]
    )
    instruction = response.output_text.strip()

    return instruction


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
