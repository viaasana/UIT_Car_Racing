from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2


from lane_detection import lane_detect
key = None

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            h, w, _ = raw_image.shape

            gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            #mySegment = lane_detect(gray_image)
            (a, b) = lane_detect(gray_image, key)


            print("left: ", a)
            print("right: ", b)


            cv2.line(raw_image, a, b, (0, 255, 0), 1)
            #cv2.line(raw_image, a, c, (0, 0, 255), 3)
            #cv2.line(raw_image, c, d, (255, 255, 0), 3)
            #cv2.line(raw_image, d, b, (0, 255, 255), 3)
            #print(state)
            cv2.imshow('raw_image', raw_image)
            #cv2.imshow('gray image', gray_image)

            # maxspeed = 90, max steering angle = 25
            AVControl(speed=10, angle=0)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
