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
    max_timesteps: int = 300
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    save_video: bool = False  # whether to save video of rollouts


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

    # TODO: 1) H5 file path
    h5_path = "/home/tennyyin/openpi-playdata/examples/droid/data/trajectories.h5"

    while True:
        count = 0
        timestamp_folder = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
        if args.save_video:
            video = []
        instruction = ""
        while count < 3:
            count += 1
            print(f"Starting rollout {count}/5...")
            # df = pd.DataFrame(columns=["success", "duration", "video_filename"])
            # Get initial observation

            # TODO: 2) Initialize empty arrays for the entire trajectory
            base_cam = []
            wrist_cam = []
            joint_position = []
            gripper_position = []
            cartesian_position = []
            action_list = []
            
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

            for t_step in bar:
                start_time = time.time()
                try:
                    # Get the current observation
                    curr_obs = _extract_observation(
                        args,
                        env.get_observation(),
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
                    if abs(curr_obs["joint_position"][6]) > 1.57:
                        env.reset()
                        break
                    env.step(action)
                    # breakpoint()

                    # TODO: 3) append to arrays
                    # curr_obs["joint_position"] - (7,)
                    # curr_obs["gripper_position"] - (1,)
                    # curr_obs["cartesian_position"] - (6,)
                    # action - (8,)
                    # base image: image_tools.resize_with_pad(curr_obs[f"{args.external_camera}_image"], 224, 224) - (224, 224, 3)
                    # wrist image_left: image_tools.resize_with_pad(curr_obs["wrist_image"]
                    # curr_obs["wrist_image"] - (720, 1280, 3), type: ndarray
                    # reshape: 180, 320, 3

                    # reshape images
                    external_image = curr_obs[f"{args.external_camera}_image"]
                    external_image_resized = cv2.resize(external_image, (320, 180))
                    wrist_image = curr_obs["wrist_image"]
                    wrist_image_resized = cv2.resize(wrist_image, (320, 180))

                    # append to arrays
                    base_cam.append(external_image_resized)
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
            # success = None
            # while not isinstance(success, float):
            #     success = input(
            #         "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            #     )
            #     if success == "y":
            #         success = 1.0
            #     elif success == "n":
            #         success = 0.0

            #     success = float(success) / 100
            #     if not (0 <= success <= 1):
            #         print(f"Success must be a number in [0, 100] but got: {success * 100}")

            # df = df.append(
            #     {
            #         "success": success,
            #         "duration": t_step,
            #         "video_filename": save_filename,
            #     },
            #     ignore_index=True,
            # )
            
            
            print("Resetting the environment for the next rollout...")
            os.makedirs("/home/tennyyin/openpi-playdata/examples/droid/results", exist_ok=True)
            #env.reset()
            with open("/home/tennyyin/openpi-playdata/examples/droid/results/observations_" + timestamp_folder + ".csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([instruction])
            # Append to CSV file

            # TODO: 4) convert to np.ndarrays
            base_cam = np.stack(base_cam)
            wrist_cam = np.stack(wrist_cam)
            joint_position = np.stack(joint_position)
            gripper_position = np.stack(gripper_position)
            cartesian_position = np.stack(cartesian_position)
            action_list = np.stack(action_list)

            # TODO: 5) Open the file each loop (append mode)
            with h5py.File(h5_path, "a") as f:
                traj_name = f"trajectory_{count}"

                # Create a new group for this trajectory
                grp = f.create_group(traj_name)

                # Create datasets for each array
                # compression="gzip"
                grp.create_dataset("base_cam", data=base_cam)
                grp.create_dataset("wrist_cam", data=wrist_cam)
                grp.create_dataset("joint_position", data=joint_position)
                grp.create_dataset("gripper_position", data=gripper_position)
                grp.create_dataset("cartesian_position", data=cartesian_position)
                grp.create_dataset("action", data=action_list)

                # add metadata as attributes
                grp.attrs["trajectory_index"] = count

        timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
        if args.save_video:
                video = np.stack(video)
                folder_path = "/home/tennyyin/openpi-playdata/examples/droid/play_videos_miyu/play_data"
                os.makedirs(folder_path, exist_ok=True)
                save_filename = folder_path + "/" + timestamp + '.mp4'
                image_clip = ImageSequenceClip(list(video), fps=10)
                image_clip.write_videofile(save_filename, codec="libx264")
        break
    # os.makedirs("results", exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    # csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    # df.to_csv(csv_filename)
    # print(f"Results saved to {csv_filename}")


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

def generate_prompt(image_right, image_wrist, object):
    base64_image_right = generate_image(image_right)
    base64_image_wrist = generate_image(image_wrist)
    # user_input = "You are given an image from a robot's right external camera and the wrist camera." \
    # "Look at the objects in the image, and separate them into categories of ""in the bowl"" and ""on the table"". " \
    # "Name them by name (e.g., apple, banana, cup, etc.) and not by color. " \
    # "Output the two lists, with elements in the list separated by commas and the list themselves separated by a "";""."\
    # "Output the two lists with no other information. "
    user_input = "You are a robot that is trying to randomly manipulate/arrange objects on the table. Existing objects on the table are: banana, carrot, green peppe, tomato."\
    "You can only choose from one of the three following task types:"\
    "pick up the <> and put it into the bowl"\
    "take the <> out of the bowl and put it on the table"\
    "push the <> towards the <left/right/up/down>"\
    "choose one object randomly. if it is in the green bowl, choose the section option. else, choose one task randomly. and fill in the <>."\
    "Only output the concise task, and do not include any other explanations."
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = [
            #{"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_input},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image_right}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image_wrist}"}
            ]}
        ]
    )
    # lists = response.output_text.strip().split(";")
    # print(lists)
    # bowl_objects = lists[0].split(":")[1].split(",")
    # table_objects = lists[1].split(":")[1].split(",")
    # bowl_objects = [obj.strip() for obj in bowl_objects]
    # table_objects = [obj.strip() for obj in table_objects]
    # if bowl_objects == ['none'] or bowl_objects == ['']:
    #     object = table_objects[np.random.randint(0, len(table_objects))].strip()
    #     random_number = np.random.randint(1, 2)
    #     if random_number == 1:
    #         prompt = f"Pick up the {object} from the table and place it in the bowl."
    #     else:
    #         prompt = f"Push the {object} on the table to the bowl."
    # elif table_objects == ['none'] or table_objects == ['']:
    #     object = bowl_objects[np.random.randint(0, len(bowl_objects))].strip()
    #     prompt = f"Pick up the {object} from the bowl and place it on the table."
    # else:
    #     random_number = np.random.randint(1, 4)
    #     if random_number == 1:
    #         object = bowl_objects[np.random.randint(0, len(bowl_objects))].strip()
    #         prompt = f"Pick up the {object} from the bowl and place it on the table."
    #     elif random_number == 2:
    #         object = table_objects[np.random.randint(0, len(table_objects))].strip()
    #         prompt = f"Pick up the {object} from the table and place it in the bowl."
    #     elif random_number == 3:
    #         object = table_objects[np.random.randint(0, len(table_objects))].strip()
    #         prompt = f"Push the {object} on the table to the bowl."
    #     else:
    #         prompt = f"Push the bowl."
    return response.output_text.strip()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
