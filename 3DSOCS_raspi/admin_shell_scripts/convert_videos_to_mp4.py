import os
import pandas as pd
import numpy as np
import cv2
from datetime import datetime, timedelta
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear

RAW_DATA_PATH= "/your/raw/data/path/here"
OUTPUT_PATH = "/your/output/path/here"
focal_dates = ["YYYY_MM_DD"] #list of dates you want to export

# Expected interframe interval in microseconds for 30fps, adjust if different
expected_interval = 33333

def get_video_files(camera, focal_date):
    """Return a list of relevant video files for a given camera, date, and event timeframe."""
    dir_path = f"{RAW_DATA_PATH}/{focal_date}/{camera}"
    try:
        video_files = [file.replace('.h264', '') for file in os.listdir(dir_path) if file.endswith('.h264')]
    except Exception as e:
        print(f"Error: {e}")
        video_files = []
    finally:
        return video_files
        
def load_metadata(camera, focal_date, video_name):
    """Load video metadata from the associated .pts file."""
    meta_file = f"{RAW_DATA_PATH}/{focal_date}/{camera}/{video_name}.pts"
    metadata = pd.read_csv(meta_file, usecols=[1, 4, 5], sep=" ", names=['system_TS', 'sensor_duration', 'synthetic_time'])
    
    # Ensure the timestamps are integers
    metadata['system_TS'] = metadata['system_TS'].astype(int)  
    metadata['sensor_duration'] = metadata['sensor_duration'].astype(int) 
    metadata['synthetic_time'] = metadata['synthetic_time'].astype(int)
    
    #calculate features
    metadata['mean_approximate_time'] = (metadata['system_TS'] + metadata['synthetic_time']) / 2
    metadata['time_difference'] = (metadata['system_TS'] - metadata['synthetic_time'])
    # Calculate the mean and variance
    mean_difference = metadata['time_difference'].mean()
    variance_difference = metadata['time_difference'].std()

    # Print the results
    #print(f"Mean of time difference: {mean_difference}")
    #print(f"Standard deviation of time difference: {variance_difference}")

    
    return metadata

def frame_dropped(interframe_interval):
    """Determine if a frame was dropped."""
    return interframe_interval >= 33333*2

def extract_frames_for_video(camera, focal_date, video_file):
    """Extract video frames for a given event timeframe."""
    print(f"[INFO] Extracting video frames for event from {video_file}")
    timestamps = []
    synthetic_boolean = []
    
    video_path = f"{RAW_DATA_PATH}/{focal_date}/{camera}/{video_file}.h264"
    
    
    try:
        metadata = load_metadata(camera, focal_date, video_file)
        stream = VideoGear(source=video_path).start()
        # Create video writers for each camera.
        
        print("[INFO] Creating output folder.")
        output_folder = f"{OUTPUT_PATH}/{focal_date}/{camera}"
        os.makedirs(output_folder, exist_ok=True)
        writer = WriteGear(output=f"{output_folder}/{video_file}.mp4", **output_params)
    except RuntimeError:
        print("Stream invalid")
    except Exception as e:
        print(f"Other error: {e}")
    else:
    
        prev_TS = None  # to keep track of the previous timestamp
        
        for _, (system_TS, sensor_duration, synthetic_time, mean_approximate_time, time_difference) in metadata.iterrows():
            
            frame = stream.read()

            if frame_dropped(sensor_duration):
                # Handle multiple consecutive dropped frames
                drop_count = int(sensor_duration // expected_interval)
                
                for _ in range(drop_count):
                    frame_to_write = create_black_frame(camera)  # Placeholder for each dropped frame
                    # Adjust the synthetic timestamp for each dropped frame
                    synthetic_TS = prev_TS + expected_interval if prev_TS else system_TS
                    timestamps.append(synthetic_TS)
                    synthetic_boolean.append(True)
                    prev_TS = synthetic_TS
                    human_readable_timestamp = epoch_to_human_readable(synthetic_TS)
                    cv2.putText(frame_to_write, str(human_readable_timestamp), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    writer.write(frame_to_write)
            else:
                frame_to_write = frame
                timestamps.append(synthetic_time)
                synthetic_boolean.append(False)
                prev_TS = synthetic_time
                
                human_readable_timestamp = epoch_to_human_readable(synthetic_time)
                # Write timestamp on the frame
                cv2.putText(frame_to_write, str(human_readable_timestamp), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                try:
                    writer.write(frame_to_write)
                except Exception as e:
                    print(f"WARNING: {e}")
                    frame = create_black_frame(camera)
                    writer.write(frame)
         
        stream.stop()
        print("[INFO] Closing video writers")                
        writer.close()
        
    finally:
        return timestamps, synthetic_boolean
    

# convert epoch timestamp to human readable format
def epoch_to_human_readable(epoch_microseconds):
    if np.isnan(epoch_microseconds):
        return "NaN"
        
    else:
        dt = datetime.fromtimestamp(epoch_microseconds/1000000.0)  # convert to seconds first
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # format and slice to remove the last 3 digits of microseconds


# Function to save the aligned frames to video files.
def save_metadata(timestamps, synthetic_boolean, camera, video_file):
    print("[INFO] Saving frames to video")
    output_folder = f"{OUTPUT_PATH}/{focal_date}/{camera}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert aligned_timestamps to human-readable format
    human_readable_times = [epoch_to_human_readable(ts) if ts is not None else np.nan for ts in timestamps]
    df = pd.DataFrame(list(zip(human_readable_times, synthetic_boolean)), columns=["timestamp", "synthetic_frame"])
    df.to_csv(os.path.join(output_folder, f"{video_file}.csv"), index=False)



def determine_frame_size(camera, focal_date):
    """Determine the frame size for a given camera based on the first frame of the first video on the focal_date."""
    video_files = get_video_files(camera, focal_date)  # get all video files for the camera on the focal_date
    if video_files:
    
        try:
            video_path = f"{RAW_DATA_PATH}/{focal_date}/{camera}/{video_files[0]}.h264"
            stream = VideoGear(source=video_path).start()
            
        except Exception as e:
            print(f"Error {e}")
            frame_shape = None
        else:
            frame = stream.read()
            stream.stop()
            frame_shape = frame.shape
        finally:
            return frame_shape
            
    else:
        frame_shape = None
        return frame_shape
    
def create_black_frame(camera):
    """Creates a black frame based on the determined shape for the camera."""
    return np.zeros(frame_sizes[camera], dtype=np.uint8)

# main function to process videos for a given date.
def main(focal_date, camera):
    video_files = get_video_files(camera, focal_date)
    for _, video_file in enumerate(video_files):
        print(f"Processing event {_} of {len(video_files)}.")

        if os.path.exists(os.path.join(OUTPUT_PATH,focal_date,camera,video_file+".mp4")):
            print("Processed")
            continue
        timestamps, synthetic_boolean = extract_frames_for_video(camera, focal_date, video_file)
        save_metadata(timestamps, synthetic_boolean, camera, video_file)

if __name__ == "__main__":
    
    camera_names = ["your_camera_names_here"]
    output_params = {
        "-c:v": "libx264", 
        "-crf": 6,
        "-input_framerate": 30
    }
    for focal_date in focal_dates:
        print(focal_date)
        frame_sizes = {camera: determine_frame_size(camera, focal_date) for camera in camera_names}
        for camera in camera_names:
            main(focal_date, camera)

