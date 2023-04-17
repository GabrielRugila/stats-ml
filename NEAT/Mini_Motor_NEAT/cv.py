import numpy as np
from PIL import ImageGrab
import cv2
import time
from directKeys import PressKey, ReleaseKey, W, A, S, D

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(img):
    original_img = img
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    return processed_img

# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)

def main():
    last_time = time.time()

    while True:
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,450)))
        new_screen = process_img(screen)
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        
        # print('down')
        # PressKey(W)
        # time.sleep(3)
        # print('up')
        # ReleaseKey(W)

        cv2.imshow('window', new_screen)

        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()