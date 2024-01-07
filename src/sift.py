import cv2
import numpy as np


def sift(img1, img2, display=False):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # Brute-Force Matching

    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test


    # cv2.drawMatchesKnn expects list of lists as matches.
    if display:
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, all_matches, None, flags=2)
        cv2.imshow('img3', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

  
    pts1 = []
    pts2 = []
    for i, (m,n) in enumerate(all_matches):
        if m.distance < 0.75*n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # # convert matches to numpy array with shape (n, 4)

    return pts1, pts2


    # matches = np.hstack((kp1[matches[:, 0]], kp2[matches[:, 1]]))
    # # remove duplicate points
    # unique = np.unique(matches, axis=0)
    # if unique.shape[0] != matches.shape[0]:
    #     print(f'{matches.shape[0]-unique.shape[0]} duplicate point(s) removed')
    #     matches = unique
    # return matches


if __name__ == '__main__':
    img1 = cv2.imread(
        '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/im1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(
        '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/im2.jpg', cv2.IMREAD_GRAYSCALE)

    pts1,pts2 = sift(img1, img2, display=True)
    
 