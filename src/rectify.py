import cv2
import numpy as np


def rectify(img1, img2, F, pts1, pts2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(F.shape)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    H1_inv = np.linalg.inv(H1)
    H2_inv = np.linalg.inv(H2)
    F_rectified = np.dot(H2_inv.T, np.dot(F, H1_inv))

    return img1_rectified, img2_rectified, F_rectified


def draw_lines(lines_set, image, points):
    for line, pt in zip(lines_set, points):
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [image.shape[1]-1, -line[2]/line[1]])
        img = cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1)
    return img


def epipolar_lines(pts1_, pts2_, F, image1, image2):
    lines1_, lines2_ = [], []
    for i in range(len(pts1_)):
        p1 = np.array([pts1_[i, 0], pts1_[i, 1], 1])
        p2 = np.array([pts2_[i, 0], pts2_[i, 1], 1])
        lines1_.append(np.dot(F.T, p2))
        lines2_.append(np.dot(F, p1))
    # draw only 10 lines
    lines1_ = np.array(lines1_)[:20]
    lines2_ = np.array(lines2_)[:20]
    
    img1 = draw_lines(lines1_, image1, pts1_)
    img2 = draw_lines(lines2_, image2, pts2_)

    out = np.hstack((img1, img2))
    cv2.imwrite("epipolar_lines.png", out)
    return lines1_, lines2_


if __name__ == '__main__':
    from sift import sift
    from ransac import ransac
    img1 = cv2.imread(
        '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/im1.jpg', 0)
    img2 = cv2.imread(
        '/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/im2.jpg', 0)
    pts1,pts2 = sift(img1, img2, display=False)
    pts1, pts2, F = ransac(pts1, pts2)
    F = F/F[2, 2]
    print(F)

    # F1, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]
    # print(F1)

    img1_rectified, img2_rectified, F_rectified = rectify(
        img1, img2, F, pts1, pts2)
    lines1_, lines2_ = epipolar_lines(
        pts1, pts2, F_rectified, img1_rectified, img2_rectified)
    cv2.imshow('img1', img1_rectified)
    cv2.imshow('img2', img2_rectified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
