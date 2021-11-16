import numpy as np
import cv2
import imutils

#----------------------------------------------------------------------------


def img_horizontal_orientation(coords):
    """
    Find horizontal tops orientation:
    rule 1st - left tops will be smallest at x
    rule 2d  - bottom tops will be biggest at y
    1st - sort at all
    2d - sort first part, sort second part and revers it,
    then concat 2 parts.
    """

    res = []

    for i in coords:

        sorted_x = i[np.argsort(i[:, 0]), :]
        top = sorted_x[:2][np.argsort(sorted_x[:2][:, 1]), :]
        bottom = sorted_x[2:][np.argsort(sorted_x[2:][:, 1]), :][::-1]

        res.append(np.concatenate((top, bottom)))

    return res


def contours_process(image, cnts, photocard, perifilter):
    """
    Get contours to list of numpy ndarrays

    perifilter - filtering figures by perimeter, small it misses
    """

    total = 0
    res = []

    for c in cnts:
        # -- Contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.08 * peri, True)

        # -- We are looking for figures which has 4 tops (rectanges)
        if len(approx) == 4 and peri > perifilter:
            total += 1
            res.append(approx)

    return res


def preparing_image(image, hsv_min, hsv_max):
    """
    Frame preparations for work, and contours capture
    """

    # -- Setting image to hsv matrix -----------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -- Filtering by color --------------------------------------------------
    edged = cv2.inRange(hsv, hsv_min, hsv_max)

    # -- Erodion
    eroded = cv2.erode(edged, None, iterations=1)

    # -- Dilation
    dilated = cv2.dilate(eroded, None, iterations=1)

    # -- Get the contours ----------------------------------------------------
    cnts = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts


def rework(image, hsv_min, hsv_max, photocard, perifilter, resolution):

    cnts = preparing_image(image, hsv_min, hsv_max)
    coords = np.squeeze(contours_process(image, cnts, photocard, perifilter))
    coords = img_horizontal_orientation(coords)
    #-- order of rectangles from left to right
    coords = np.sort(coords,axis=0)
    

    # -- img preparation block
    h, w = photocard.shape[:2]
    photo_coords = np.float32([([[0, 0]]), ([[0, h]]), ([[w, h]]), ([[w, 0]])])

    # -- warp image to contour
    h, mask = cv2.findHomography(photo_coords, coords[-1])
    warped = cv2.warpPerspective(photocard, h, resolution,
                                 flags=cv2.INTER_LINEAR)

    # -- create black background, for normal replacing
    cv2.fillPoly(image, pts=[coords[-1]], color=(0, 0, 0))

    # -- photocard pushing
    image = cv2.addWeighted(image, 1, warped, 1, 0)

    return image

#----------------------------------------------------------------------------


if __name__ == '__main__':

	# -- HSV colors capture matrixes
    hsv_min_1 = np.array((39, 80, 56), np.uint8)
    hsv_max_1 = np.array((85, 255, 255), np.uint8)

    # -- Video, photo capture
    video_capture = cv2.VideoCapture('video.mp4')
    photocard = cv2.imread('photo.jpg')

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('OUT555K.mp4', fourcc,
                          video_FPS, (frame_width, frame_height))

    try:
        for i in range(0, frame_count):

            print(i)

            ret, image = video_capture.read()

            FRAME = rework(image, hsv_min_1, hsv_max_1, photocard,
            				150, (frame_width,frame_height))
    
            out.write(FRAME)
    except:
        out.release()
        cv2.destroyAllWindows()

    out.release()
    cv2.destroyAllWindows()
