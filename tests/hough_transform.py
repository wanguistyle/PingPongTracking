import sys
import math
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = "data/philo_ligne_crop.png"
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    height, width = src.shape
    roi_start_row = int(height * 0.45)
    
    roi = src[roi_start_row:height, 0:width]

    dst_roi = cv.Canny(roi, 50, 200, None, 3)
    
    cdstP = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    
    linesP = cv.HoughLinesP(dst_roi, 1, np.pi / 180, 50, None, 200, 80)
    
    # Dessiner les lignes en réajustant l'offset Y
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1] + roi_start_row), 
                           (l[2], l[3] + roi_start_row), 
                           (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines - Bottom Only", cdstP)
    dst = cv.Canny(src, 50, 200, None, 3)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])
