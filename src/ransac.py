# import numpy as np
# import cv2

import numpy as np
import cv2
from math import log, log1p, pow


def ransac(pt1, pt2):
    n_rows = np.array(pt1).shape[0]

    # RANSAC
    best_F = None
    best_inliers = 0
    n = 0
    Niter = 1000
    beta = 0.01
    threshold = 1.5

    while (n < Niter):
        indices = []
        # randomly select 8 points
        random = np.random.choice(n_rows, size=8)
        img1_8pt = pt1[random]
        img2_8pt = pt2[random]
        F = get_F_matrix(img1_8pt, img2_8pt)

        for j in range(n_rows):
            x1 = pt1[j]
            x2 = pt2[j]

            # error computation
            pt1_ = np.array([x1[0], x1[1], 1])
            pt2_ = np.array([x2[0], x2[1], 1])
            d = np.abs(np.dot(pt1_.T, np.dot(F, pt2_))) / \
                np.sqrt(np.dot(F, pt2_)[0]**2 + np.dot(F, pt2_)[1]**2)

            if np.abs(d) < threshold:
                indices.append(j)

        if len(indices) > best_inliers:
            best_inliers = len(indices)
            final_indices = indices
            best_F = F
            Niter = log(beta)/log1p(-pow(best_inliers/n_rows, 8))
            print(n, "/", Niter, " | inliers:", best_inliers)
        n += 1

    img1_points = pt1[final_indices]
    img2_points = pt2[final_indices]

    return img1_points, img2_points, best_F


def normalize_points(pts):
    """
    This function normalizes the points
    so that the origin is at centroid and
    mean distance from origin is sqrt(2).
    """

    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])
    x = pts[:, 0] / mean_x
    y = pts[:, 1] / mean_y
    pts = np.vstack((x, y)).T
    N = np.array([[1 / mean_x, 0, 0], [0, 1 / mean_y, 0], [0, 0, 1]])

    return pts, N


def get_F_matrix(pts1, pts2, normalize=True):
    """
    This function returns the fundamental matrix F
    given two sets of corresponding points pts1 and pts2
    """

    # Convert to homogeneous coordinates
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    if normalize:
        pts1, N1 = normalize_points(pts1)
        pts2, N2 = normalize_points(pts2)

    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    # Construct the A matrix
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 *
                  y1, y2, x1, y1, np.ones(x1.shape))).T

    # Solve for the SVD
    U, S, V = np.linalg.svd(A)

    # The fundamental matrix is the last column of V
    F = V[-1].reshape(3, 3)

    # Enforce rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    # Denormalize
    if normalize:
        F = N2.T @ F @ N1

    return F


def drawlines(img1, img2, lines, pts1, pts2):
    """
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c = img1.shape[:2]
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())  # random color

        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1[:2].astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2[:2].astype(int)), 5, color, -1)

    return img1, img2


if __name__ == "__main__":
    from sift import sift
    import matplotlib.pyplot as plt
    # read tif images
    img1 = cv2.imread('/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/run-L.jpeg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/Users/newt/Desktop/CS/SDI/VIC/Project/src/data/run-R.jpeg', cv2.IMREAD_GRAYSCALE)


    pts1, pts2 = sift(img1, img2, display=False)

    pt1, pt2, F = ransac(pts1, pts2)

    # CV2 LMEDS for comparison
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # pt1 = pts1[mask.ravel() == 1]
    # pt2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)[:100]
    img5, img6 = drawlines(img1, img2, lines1, pt1, pt2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)[:100]
    img3, img4 = drawlines(img2, img1, lines2, pt2, pt1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
