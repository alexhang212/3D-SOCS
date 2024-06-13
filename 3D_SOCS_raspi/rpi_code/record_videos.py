"""
Records time synchronized videos with custom encoder on follower RPI CM4
Author: Michael Chimento
"""

import RPi.GPIO as GPIO
import threading
import cv2
import signal
from time import sleep, time_ns, strftime
from datetime import datetime
from sys import exit
from os import uname
from picamera2.encoders import H264Encoder, Quality
from picamera2 import Picamera2
from libcamera import controls, Transform

INPUT_PIN = 5
DESIRED_FRAMERATE = 30
FRAME_DURATION_US = int(1e6 / DESIRED_FRAMERATE)
LENS_POSITION = 0.52
TRANSFORM = Transform(hflip=0,vflip=0)
WIDTH = 1280
HEIGHT = 720

ZONE1 = 5000
ZONE2 = 100
BIG_STEP_SIZE = 500
SMALL_STEP_SIZE = 100
BUFFER_MAX_SIZE = 1000
data_buffer = []
lock = threading.Lock()

camera_name = uname()[1]
synthetic_time = 0
previous_sensor_TS = 0
previous_system_TS = 0
frame_idx = 0
l_frames=1 
shit_frames=1
video_idx=1

colour = (255, 255, 0)
origin = (530, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2

def buffered_data_writer():
    if len(data_buffer) > 0:
        # Use the current_output from the first item in the buffer for all items.
        for item in data_buffer:
            output_file = item[0][:-4]+"pts"
            write_data_to_file(output_file, item[1])
        data_buffer.clear()

def write_data_to_file(output_file, data_buffer):
    with open(output_file, "a") as f:
        f.write(f"{data_buffer[0]} {data_buffer[1]} {data_buffer[2]} {data_buffer[3]} {data_buffer[4]} {data_buffer[5]} {data_buffer[6]}\n")
        
def close_buffer():
    print("Cleaning up buffer.")
    if len(data_buffer) > 0:
        for item in data_buffer:
            output_file = item[0][:-4]+"pts"
            write_data_to_file(output_file, item[1])
        data_buffer.clear()
    print("Buffer written and cleared.")

def print_status(status, correction, sensor_duration, ratio, time_diff):
    print(f"{status} | RequestedDuration: {correction} | FrameDuration: {sensor_duration} | Ratio: {ratio} | Offset: {time_diff} ", end='\r', flush=True)

def signal_handler(sig, frame):
    print("Closing camera.")
    if camera.is_open:
        camera.close()
    close_buffer()
    GPIO.cleanup()
    exit(0)

def rising_edge_callback(channel):
    global video_idx, current_output, frame_idx, synthetic_time, previous_sensor_TS, previous_system_TS
    
    sleep(0.1)
    if GPIO.input(INPUT_PIN):
        reset_global_vars()
        start_recording()
        stop_recording()
        buffered_data_writer()
        video_idx += 1

def reset_global_vars():
    global synthetic_time, previous_sensor_TS, previous_system_TS, frame_idx
    synthetic_time = 0
    previous_sensor_TS = 0
    previous_system_TS = 0
    frame_idx = 0
    
def start_recording():
    global current_output, camera, encoder, video_idx
    current_output = f"Videos/{camera_name}_{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}_video_{video_idx}.h264"
    camera.start_recording(encoder=encoder, output=current_output, quality=Quality.HIGH)
    print(f"Recording started video {video_idx} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}")

def stop_recording():
    global camera, video_idx
    wait_for_input_fall()
    camera.stop_recording()
    print(f"Recording stopped video {video_idx} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}")

def wait_for_input_fall():
    input_fell = 0
    while True:
        if not GPIO.input(INPUT_PIN):
            input_fell += 1
        else:
            input_fell = 0
        if input_fell > 5:
            break
            
def setup_gpio():
    print(f"Setting up GPIO. Input pin assigned: {INPUT_PIN}")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def setup_camera_and_encoder():
    print(f"Setting up camera and recorder. Lens position={LENS_POSITION}")
    camera = Picamera2()
    video_config = camera.create_video_configuration(
        main = {"size":(WIDTH,HEIGHT)},
        transform = TRANSFORM,
        controls={"NoiseReductionMode": 1,
                  "FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US),
                  "AfMode": controls.AfModeEnum.Manual,
                  "LensPosition": LENS_POSITION})
    camera.configure(video_config)
    encoder = H264Encoder(enable_sps_framerate=True)
    camera.pre_callback = discipline_framerate
    camera.start()
    return camera, encoder

def setup_gpio_event():
    GPIO.add_event_detect(INPUT_PIN, GPIO.RISING, callback=rising_edge_callback, bouncetime=300)

def discipline_framerate(request):
    global frame_idx, l_frames, shit_frames, synthetic_time, previous_sensor_TS, previous_system_TS
    metadata = request.get_metadata()
    
    if metadata:
        
        system_TS = int(time_ns() / 1000)
        system_duration = system_TS - previous_system_TS
        previous_system_TS = system_TS
        
        sensor_TS = int(metadata['SensorTimestamp'] / 1000)
        sensor_duration = (sensor_TS - previous_sensor_TS)
        previous_sensor_TS = sensor_TS
        
        if frame_idx==0:
            system_duration = 0
            sensor_duration = 0

        if not synthetic_time:
            synthetic_time = system_TS
        else:
            synthetic_time += sensor_duration

        raw_time_diff = synthetic_time % FRAME_DURATION_US
        
        if raw_time_diff < (FRAME_DURATION_US / 2):
            time_diff = raw_time_diff
        else:
            time_diff = - (FRAME_DURATION_US - raw_time_diff)
            
        fraction = abs(time_diff) / ZONE1
        
        if time_diff < 0:
            framerate_cor = BIG_STEP_SIZE if time_diff < -ZONE1 else (SMALL_STEP_SIZE * fraction)
        else:
            framerate_cor = -BIG_STEP_SIZE if time_diff > ZONE1 else (-SMALL_STEP_SIZE * fraction)
        
        correction = int(FRAME_DURATION_US + framerate_cor)
        current_FD = camera.camera_controls["FrameDurationLimits"]
        
        if current_FD is not correction:
            camera.set_controls({'FrameDurationLimits': (correction, correction)})
            
        locked = abs(time_diff) < 1000
        
        if encoder.output and encoder.output.recording:
            line = (frame_idx, system_TS, sensor_TS, system_duration, sensor_duration, synthetic_time, locked)
            #add data to buffer
            data_buffer.append((current_output, line))
        
        #ratio = l_frames / (l_frames + shit_frames)
        
        #if abs(time_diff) < 1000:
        #    l_frames += 1
        #    print_status("LOCKED", correction, sensor_duration, ratio, time_diff)
        #else:
        #    shit_frames += 1
        #    print_status("######", correction, sensor_duration, ratio, time_diff)
    frame_idx += 1

if __name__ == "__main__":
    writer_thread = threading.Thread(target=buffered_data_writer)
    writer_thread.start()
    setup_gpio()
    camera, encoder = setup_camera_and_encoder()
    setup_gpio_event()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.pause()


