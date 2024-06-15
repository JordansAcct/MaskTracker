#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ------------ DEPENDENCIES AND UTILITIES --------------


# In[23]:


import sys
sys.path.insert(0,'/home/xilinx/jupyter_notebooks/mask_tracker/caffe/distribute/python/')

import os
import time
import cv2 
import numpy as np
import socket
import caffe

from threading import Thread, BoundedSemaphore
# vvv old face detection method
#from face_recognition import load_image_file as load_face_rec_img, face_locations
from matplotlib import image, pyplot as plt
from PIL import Image
from time import sleep
from math import ceil

from pynq import Overlay, allocate
from pynq.ps import Clocks
from pynq.lib import Wifi
from pynq.lib.video import *
from pynq.lib.pmod.pmod_als import Pmod_ALS

from driver_base import FINNExampleOverlay as finn_overlay
from driver import io_shape_dict as isd

from facedetect.caffe.caffe_inference import inference as face_inference  

accel = finn_overlay(bitfile_name="resizer.bit", platform="zynq-iodma", io_shape_dict=isd)
model_dir = "./facedetect/caffe/model/Slim-320"  # you will have to rename the lightweight face detection directory


global_cam = cv2.VideoCapture(0)  # extremely annoying if not shut down properly so took it out of thread
face_net = caffe.Net(model_dir + "/slim-320.prototxt", model_dir + "/slim-320.caffemodel", caffe.TEST)

cam_lock = BoundedSemaphore(value=1)
face_lock = BoundedSemaphore(value=1)
classify_lock = BoundedSemaphore(value=1)

cam_buf = []
face_buf = []
classify_buf = []  # [0] = INCORRECT, [1] = CORRECT, [2] = NONE

kill_pipeline = False
kill_program = False
global_verbose = True


# ----------- CONNECTION RELATED UTILITIES -----------------
class wifi_util():
    def __init__ (self, ssid: str, password: str) -> None:
        self.ssid = ssid
        self.password = password
        self.is_connected = False
        self.port = Wifi()
    
    # set one or both of ssid/password to new value
    def set_credentials(self, ssid: str = None, password: str = None):
        if ssid:
            self.ssid = ssid
        if password:
            self.password = password
    
    # return info about wifi/network params
    def info(self) -> str:
        status = "DOWN"
        if (self.is_connected):
            status = "UP"
            
        return (f"Wifi status:    {status}\n"
                f"SSID = {self.ssid}\n"
                f"PASS = {self.password}\n")
    
    # connect to network with existing or new ssid/password. Use force to kill an existing connection
    def connect(self, ssid: str = None, password: str = None, force: bool = False) -> bool:
        if force:
            self.disconnect()
            
        if (self.is_connected):
            return False
        
        curssid = self.ssid
        curpass = self.password
        if ssid:
            curssid = ssid
        if password:
            curpass = password
        
        self.port.connect(curssid, curpass, force)
        self.is_connected = True
        
        return self.is_connected
    
    # disconnect from network if connected
    def disconnect(self) -> bool:
        if (not self.is_connected):
            return False
        
        self.port.reset()
        
        self.is_connected = False
        return True


# ----------- CAMERA RELATED UTILITIES --------------
class camera():  # TODO: USELESS 
    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = global_cam

    def cam_up(self) -> None:
        # self.cap.set(CV_CAP_PROP_BUFFERSIZE, 1); # internal buffer will now store only 1 frames
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Check if camera opened successfully
        if (self.cap.isOpened() == False): 
            print("Unable to read camera feed")
        else:
            print("Camera opened successfully")
            
    def cam_down(self) -> None:
        self.cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        
def print_classifications(arr):
    ret = []
    for item in arr:
        if item == 0:
            ret.append("incorrect_mask")
        elif item == 1:
            ret.append("correct_mask")
        elif item == 2:
            ret.append("no mask")
        else:
            ret.append("unknown")
    return ret


# -------------- THREADS ---------------

class template_thread(Thread):
    def __init__(self, thread_id: int = 0, name: str = "unnamed thread", 
                 verbose: bool = global_verbose, kill: bool = kill_pipeline) -> None:
        Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.verbose = verbose
        self.kill = kill  # TODO: get rid of this i was too lazy to do it now sorry

        
# -------------- VIDEO READING THREAD ------------
class video_reader(template_thread):
    def __init__(self, verbose: bool = global_verbose, show_imgs: bool = False, kill: bool = kill_pipeline):
        template_thread.__init__(self, 1, "video read thread", verbose, kill)

        self.show_imgs = show_imgs
        
    def run(self) -> None:
        if self.verbose: print(f"Running {self.name} [ID: {self.thread_id}].")
            
        global cam_buf
        
        while not kill_pipeline:
            for i in range(4):
                success, frame = global_cam.read()  # FIXME: buffering/huge delay
            
            if success and frame is not None:
                if self.show_imgs:
                    plt.imshow(frame[:,:,[2,1,0]])
                    plt.show()  
                
                cam_buf = [frame]
                try:  # ValueError if lock value is 1, but we don't care!
                    cam_lock.release()
                except ValueError: 
                    pass

            else:
                if self.verbose: print(f"Couldn't read image...")
            
        if self.verbose: print(f"Received kill signal. Exiting '{self.name}'...")

            
# ------------- DETECTION THREAD ------------
class face_detect(template_thread):
    def __init__(self, verbose: bool = global_verbose, 
                 show_imgs: bool = False, kill: bool = kill_pipeline):
        template_thread.__init__(self, 2, "face detect thread", verbose, kill)
        
        self.show_imgs = show_imgs
        self.input_size = (320, 240)
    
    def run(self):
        if self.verbose: print(f"Running {self.name} [ID: {self.thread_id}].")
        
        global cam_buf
        global face_buf
        global kill_pipeline
        
        while not kill_pipeline:
            
            if cam_lock.acquire(timeout=5):
                img = cam_buf[0] 
            else:
                if self.verbose: print(f"{self.name}: timout trying to acquire cam_lock")
                continue
                
            #img = cv2.resize(img, self.input_size, cv2.INTER_NEAREST)
                
            if self.show_imgs and self.verbose: 
                plt.imshow(img[:,:,[2,1,0]])
                plt.show() 
                
            start = time.time()  # BEGIN TIMING 
            #in_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #face_coords = face_locations(in_img)       
            
            boxes, labels, probs, out_img = face_inference(img, face_net, self.show_imgs)
            
            end = time.time()  # END TIMING
            runtime = end - start
            if self.verbose: 
                print(f"{self.name}: runtime(ms): {runtime * 1000}")
            
            faces = []
                            
            for box in boxes:
                face_crop = img[box[1]:box[3], box[0]:box[2]]
                faces.append(face_crop)
            
            if self.show_imgs:   
                plt.imshow(out_img)
                plt.show() 
                if self.verbose:
                    for face in faces:
                        plt.imshow(face[:,:,[2,1,0]])
                        plt.show()  
            
                
            face_buf = faces 
            try:
                face_lock.release()
            except ValueError: 
                pass
            
        if self.verbose: print(f"Received kill signal. Exiting '{self.name}'...")

            
# ------------- IMAGE PROCESSING + CLASSIFY THREAD ----------
class image_processor(template_thread):
    def __init__(self, in_x: int, in_y: int, 
                 verbose: bool = global_verbose, kill: bool = kill_pipeline) -> None:
        template_thread.__init__(self, 3, "classify thread", verbose, kill)
        
        self.in_x = in_x
        self.in_y = in_y     
        
    def flatten_img(self, img, x_axis: int, y_axis: int = 0):
        if not y_axis:  # Specify only 1 dimension if result is square
            y_axis = x_axis
            
        assert x_axis > 0 and y_axis > 0, f"x and y axes should have positive value, got ({x_axis}, {y_axis}) instead."
            
        if x_axis != y_axis:
            # CROP IMAGE TO CENTER 
            width, height = img.size()
            smaller = width
            if height < width: 
                smaller = height

            left = (width - smaller)/2
            top = (height - smaller)/2
            right = (width + smaller)/2
            bottom = (height + smaller)/2

            img = img.crop((left, top, right, bottom))
            
        img = cv2.resize(img, (x_axis, y_axis), cv2.INTER_NEAREST) # Using nearest-neighbor for better performance
        img = img.reshape(1, x_axis, y_axis, 3)
        
        return img
        
    def run(self) -> None:
        if self.verbose: print(f"Running {self.name} [ID: {self.thread_id}].")
            
        global face_buf
        global classify_buf
        global kill_pipeline
            
        while not kill_pipeline:
            
            if face_lock.acquire(timeout=5):
                imgs = face_buf  # Grab images from buffer
            else:
                if self.verbose: print(f"{self.name}: timed out trying to acquire face_lock...")
                continue
            
            # [0] = INCORRECT, [1] = CORRECT, [2] = NONE
            classifications = [0, 0, 0]

            start = time.time()
            for img in imgs:
                img = self.flatten_img(img, self.in_x, self.in_y)
                result = accel.execute(img)  # Will return top 2 classifications (out of 3)
                try: 
                    classifications[int(result[0][0])] += 1
                except:
                    print(f"unknown classification {result}, {result[0]}, {result[0][0]} was returned.")
            end = time.time()
            runtime = end - start
            if self.verbose: 
                print(f"{self.name}: runtime(ms): {runtime * 1000}, throughput[images/s]: {len(imgs) / runtime}")
                print(f"classifications: incorrect: {classifications[0]}, correct: {classifications[1]}, none: {classifications[2]}")
                
                
            classify_buf = classifications  # does not wait for classifications to be consumed
            try:
                classify_lock.release()
            except ValueError: 
                pass
            
        if self.verbose: print(f"Received kill signal. Exiting '{self.name}'...")

            
# ------------- DATA TRANSMISSION THREAD -----------
class send_results(template_thread):
    def __init__(self, ssid: str, pwd: str, 
                 verbose: bool = global_verbose, kill: bool = kill_pipeline) -> None:
        template_thread.__init__(self, 4, "transmission thread", verbose, kill)
        
        self.ssid = ssid
        self.pwd = pwd
        self.port = wifi_util(self.ssid, self.pwd)
        
        self.dest_addr = ("10.16.230.118", 12000)
        
        
    def send_data(self, data) -> None:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.settimeout(1.0)

        message = str(data[1]).zfill(2) + str(data[0]).zfill(2) + str(data[2]).zfill(2)
        if self.verbose: print(f"Sending message '{message}'...")
        client_socket.sendto(message.encode(), self.dest_addr)
            
    
    def run(self) -> None:
        if self.verbose: print(f"Running {self.name} [ID: {self.thread_id}].")
        self.port.connect()
        
        if self.verbose: print(f"Wifi status:\n{self.port.info()}")
        
        global classify_buf
        global kill_pipeline
        
        while not kill_pipeline:

            if classify_lock.acquire(timeout=5):
                data = classify_buf 
            else:
                if self.verbose: print(f"{self.name}: timout trying to acquire classify_lock")
                continue


            if not (self.send_data(data)):
                if self.verbose: print(f"An error occured while sending the following data: {data}")
            else:
                if self.verbose: print(f"sent {data}.")
        
        if self.verbose: print(f"Received kill signal. Exiting '{self.name}'...") 
        self.port.disconnect()

        
class keyboard_reader(Thread):
    def __init__(self, name='keyboard thread', verbose: bool = global_verbose):
        super(keyboard_reader, self).__init__(name=name, daemon=True)
        self.verbose = verbose
        self.start()

    def run(self):
        global kill_program
        
        inp = input()  # block until user input
        if self.verbose: print(f"{self.name}: Got user input '{inp}', exiting program...")
        kill_program = True


# ----------- MAIN ------------------
def main():
    print("Starting.")

    verbose = True
    global kill_program
    global kill_pipeline
        
    is_dark: bool = False  # FIXME: should be true once ALS is working
    # kill_als = False
    # reader will be present for entire process lifetime, so treat it differently than other threads
    # if verbose: print("Creating ALS thread...")
    # als = ALS_reader(status=ALS_stat, verbose=verbose, kill=kill_als)
    is_asleep = True  # start sleeping, no benefit to assuming awake
        
    pipeline = []
    
    keyboard_reader()  # stop program on any keyboard input
    
    while not kill_program:
        # TODO: remove busy waiting and actually sleep

        if (is_asleep and not is_dark):  # if asleep, check for wake up
            cam_lock.acquire(timeout=0.01)  # if value is 1, decrement so consumers have to wait
            face_lock.acquire(timeout=0.01)
            classify_lock.acquire(timeout=0.01)
            
            # ---- WAKE ----
            kill_pipeline = False
            pipeline = []  # just in case it wasn't emptied out

            if verbose: print(f"Creating video thread...")
            pipeline.append(video_reader())

            if verbose: print(f"Creating face detect thread...")
            pipeline.append(face_detect(show_imgs=True))

            if verbose: print(f"Creating classify thread...")
            pipeline.append(image_processor(in_x=32, in_y=32))

            #if verbose: print(f"Creating transmission thread...")
            #pipeline.append(send_results(ssid="ssid", pwd="pwd"))

            for thread in pipeline:
                if verbose: print(f"Starting thread {thread.name}...")
                thread.start()

            is_asleep = False   
            
        elif (not is_asleep and is_dark):  # if awake, check for sleep
            # ---- SLEEP ----
            kill_pipeline = True  # stop all threads except ALS

            for thread in pipeline:
                thread.join()

            pipeline = []  # empty out array
            is_asleep = True


        sleep(0.1)

        
    kill_pipeline = True
    for thread in pipeline:
        thread.join()
        
    if verbose: print(f"Killed by user. Exiting main...")
    global_cam.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    print("Main.")
    main()


