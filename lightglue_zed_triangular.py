from pathlib import Path

from lightglue.superpoint import SuperPoint
from lightglue.disk import DISK

from lightglue import LightGlue
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import cv2
import torch
import numpy as np
import time

import pyzed.sl as sl
import matplotlib.pyplot as plt


# configurations
use_contour = True
save_figs = True

def find_contours(image_path, blur_kernel=(13, 13), threshold_method="adaptive", threshold_value=127,  
                  min_area=100, max_area=None, save_figs=True):
    """
    Finds contours in an image.

    Args:
        image_path: Path to the input image.
        blur_kernel: Tuple representing the kernel size for Gaussian blur.  (e.g., (5,5), (3,3)).  Helps reduce noise.
        threshold_method:  "adaptive" or "simple".  Adaptive is generally better for varying lighting.
        threshold_value:  Only used if threshold_method is "simple". The threshold value.
        min_area: Minimum area of a contour to be considered.  Helps filter out small noise.
        max_area: Maximum area of a contour to be considered. If None, no maximum area filter is applied.

    Returns:
        A tuple containing:
            - A copy of the original image with contours drawn on it.
            - A list of the contours found (as numpy arrays of points).  Can be empty.
            - A grayscale thresholded image (useful for debugging).
            - The original image (useful for debugging).
    """

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # reverse color: the detecting object is dark while the background is light
        gray = cv2.bitwise_not(gray)

        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

        # Threshold the image
        if threshold_method == "adaptive":
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adjust block size (11) and C (2) as needed
        elif threshold_method == "simple":
            # The simple threshold method is simple yet strong -- use simple for most easy-to-detect cases
            _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError("Invalid threshold_method. Choose 'adaptive' or 'simple'.")


        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Use RETR_EXTERNAL for outer contours

        # Filter contours based on area (optional, but highly recommended)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area and (max_area is None or area <= max_area): # Check min and max area
                filtered_contours.append(contour)


        # Draw contours on a copy of the original image
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 0, 255), 10)  # Red contours
        # red contours are over every contours detected
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        # find for the largest contour, which we may be finding.
        # To precisely select which object to create contour, we may use bounding boxes if needed.
        x,y,w,h = cv2.boundingRect(largest_contour)
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 0), 10)  # Green largest contour
        # green contour line is over the largest (selected) contour
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (255, 0, 0), 10)  # Blue bounding box
        # blue bounding box is over the largest (selected) contour
        if save_figs:
            cv2.imshow(img_with_contours)
            cv2.waitKey(0)
            cv2.imwrite("results/"+image_path[8:], img_with_contours)
        return img_with_contours, filtered_contours, thresh, img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)


# setup zed camera and get camera matrices
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()
if zed.grab() != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()
camconfig = zed.get_camera_information().camera_configuration.calibration_parameters
cam_intrinsics = [[camconfig.left_cam.fx, 0, camconfig.left_cam.cx],
                        [0, camconfig.left_cam.fy, camconfig.left_cam.cy],
                        [0, 0, 1]]
                # left and right camera has same camera settings: use single cam_intrinsics for both cameras.
left_extrinsic = [[1, 0, 0, -0.06],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]]
right_extrinsic = [[1, 0, 0, 0.06],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]]
P_left = np.dot(cam_intrinsics, left_extrinsic)
P_right = np.dot(cam_intrinsics, right_extrinsic)

# take image
date = time.strftime("%Y-%m-%d_%H-%M-%S")
image_zed_l = sl.Mat()
image_zed_r = sl.Mat()
zed.retrieve_image(image_zed_l, sl.VIEW.LEFT)
img_l = image_zed_l.get_data()
zed.retrieve_image(image_zed_r, sl.VIEW.RIGHT)
img_r = image_zed_r.get_data()
cv2.imwrite("results/"+date+"_l.jpg", img_l)
cv2.imwrite("results/"+date+"_r.jpg", img_r)
zed.close()


# load and extract feature points
img0 = "results/"+date+"_l.jpg"
img1 = "results/"+date+"_r.jpg"
image0 = load_image(img0)
image1 = load_image(img1)
feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
img_with_contours0, contours0, thresh0, original_image0 = find_contours(img0, min_area=10000, save_figs=save_figs)
img_with_contours1, contours1, thresh1, original_image1 = find_contours(img1, min_area=10000, save_figs=save_figs)

ti = time.time()

if use_contour & bool(contours0) & bool(contours1):

    contour0 = torch.tensor(max(contours0, key=cv2.contourArea)).to(device)
    contour1 = torch.tensor(max(contours1, key=cv2.contourArea)).to(device)
    feat0_exp = feats0['keypoints'][0].unsqueeze(1)
    feat1_exp = feats1['keypoints'][0].unsqueeze(1)
    cont0_exp = contour0.unsqueeze(0).reshape(1, -1, 2)
    cont1_exp = contour1.unsqueeze(0).reshape(1, -1, 2)
    dist0 = torch.sqrt(torch.sum((feat0_exp - cont0_exp) ** 2, dim=2))
    dist1 = torch.sqrt(torch.sum((feat1_exp - cont1_exp) ** 2, dim=2))

    min_d0, idx0 = torch.min(dist0, dim=1)
    min_d1, idx1 = torch.min(dist1, dim=1)
    close_inds0 = torch.nonzero(min_d0 < 13).squeeze(1)
    close_inds1 = torch.nonzero(min_d1 < 13).squeeze(1)

    selected_f0 = {"keypoints": [[]], "descriptors": [[]], "keypoint_scores": [], "image_size": feats0["image_size"]}
    selected_f1 = {"keypoints": [[]], "descriptors": [[]], "keypoint_scores": [], "image_size": feats1["image_size"]}
    selected_f0['keypoints'] = torch.index_select(feats0['keypoints'][0], 0, close_inds0).reshape(1,-1,2)
    selected_f0['descriptors'] = torch.index_select(feats0['descriptors'][0], 0, close_inds0).reshape(1,-1,256)
    selected_f0['keypoint_scores'] = torch.index_select(feats0['keypoint_scores'][0], 0, close_inds0).reshape(1,-1)
    selected_f1['keypoints'] = torch.index_select(feats1['keypoints'][0], 0, close_inds1).reshape(1,-1,2)
    selected_f1['descriptors'] = torch.index_select(feats1['descriptors'][0], 0, close_inds1).reshape(1,-1,256)
    selected_f1['keypoint_scores'] = torch.index_select(feats1['keypoint_scores'][0], 0, close_inds1).reshape(1,-1)

    matches01 = matcher({"image0": selected_f0, "image1": selected_f1})
    selected_f0, selected_f1, matches01 = [
        rbd(x) for x in [selected_f0, selected_f1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = selected_f0["keypoints"], selected_f1["keypoints"], matches01["matches"]
else:
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
if save_figs:
    viz2d.save_plot("results/"+date+"_matches_"+str(use_contour)+".jpg")

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([okpt0, okpt1], colors=["red", "red"], ps=6)
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
if save_figs: 
    viz2d.save_plot("results/"+date+"_keypoints_"+str(use_contour)+".jpg")
viz2d.show()

# triangulate
crd0 = kpts0[matches[..., 0]]
crd1 = kpts1[matches[..., 1]]
crd0 = crd0.cpu().numpy()
crd1 = crd1.cpu().numpy()
P0 = np.array(P_left)
P1 = np.array(P_right)

if crd0.size:
    res = cv2.triangulatePoints(P0, P1, crd0.T, crd1.T)
    # res = cv2.triangulatePoints(P0, P1, crd0, crd1)
    res_xyz = res[:3] / res[3]
    # get colors of the points from the left image
    # colors = cv2.cvtColor(image0.numpy(), cv2.COLOR_BGR2RGB)
    imgclrs = np.transpose(image0.numpy(), (1, 2, 0))
    colors = imgclrs[crd0[:, 1].astype(int), crd0[:, 0].astype(int)]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(-res_xyz[0], res_xyz[1], res_xyz[2], marker='x', color=colors)
    ax.scatter((-0.06, 0.06), (0,0), (0,0), marker='o', color='r')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax.scatter((0),(0),(0), marker='P', color='k')
    ax.plot((-0.06, 0.06), (0,0), (0,0), color='r')
    ax.plot((0,0), (-0.1, 0.05), (0,0), color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    if save_figs:
        ax.view_init(90,-90,0)
        plt.savefig("results/"+date+"_triangulation_"+str(use_contour)+"_1.jpg")
        ax.view_init(120,-45,45)
        plt.savefig("results/"+date+"_triangulation_"+str(use_contour)+"_2.jpg")

        plt.show()

    # save the results matrix
    np.save("triangulated_points1.npy", res_xyz)
    np.save("triangulated_colors1.npy", colors)
else:
    print("No keypoints found")

tf = time.time()
print(f"Time {str(use_contour)}: {tf-ti}")

cv2.waitKey(0)