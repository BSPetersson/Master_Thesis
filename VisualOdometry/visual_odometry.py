from cmath import nan
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import time
from lib.visualization import plotting
from lib.visualization import plotting_mpl
from lib.visualization.video import play_trip
import deepFeatures
import imutils
import random

from tqdm import tqdm

from matplotlib.colors import LinearSegmentedColormap


class VisualOdometry():
    def __init__(self, data_dir, show, feature_extractor='orb', nfeatures=6000):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(nfeatures)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.show = show
        self.kp2, self.des2 = (None, None)
        self.feature_extractor = feature_extractor
        self.match_count = 0

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return T


    def get_matches(self, data_dir, i, use_mean_filter, discard_std_multiplier=1.2, nfeatures=6000):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        # Find the keypoints and descriptors with ORB
        # Replace ORB feature extraction with deepFeatures feature extraction
        if self.kp2 is None:
            if self.feature_extractor == 'mobilenet_v2':
                kp1, des1 = deepFeatures.get_image_features(self.images[i - 1], model_name='mobilenet_v2', nfeatures=nfeatures)
            elif self.feature_extractor == 'vgg':
                kp1, des1 = deepFeatures.get_image_features(self.images[i - 1], model_name='vgg', nfeatures=nfeatures)
            else:
                kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        else:
            kp1, des1 = self.kp2, self.des2
        if self.feature_extractor == 'mobilenet_v2':
            self.kp2, self.des2 = deepFeatures.get_image_features(self.images[i], model_name='mobilenet_v2', nfeatures=nfeatures)
        elif self.feature_extractor == 'vgg':
            self.kp2, self.des2 = deepFeatures.get_image_features(self.images[i], model_name='vgg', nfeatures=nfeatures)
        else:
            print("images: {}".format(len(self.images)))
            self.kp2, self.des2 = self.orb.detectAndCompute(self.images[i], None)

        # Find matches
        matches = self.flann.knnMatch(des1, self.des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        val = 0.8
        while len(good) < 8 and val < 1.0:
            good = []
            try:
                for m, n in matches:
                    if m.distance < val * n.distance:
                        good.append(m)
            except ValueError:
                pass
            val = val + 0.01

        draw_params = dict(matchColor = -1, # draw matches in green color
                singlePointColor = None,
                matchesMask = None, # draw only inliers
                flags = 2)

        img3 = cv2.drawMatches(self.images[i-1], kp1, self.images[i], self.kp2, good ,None,**draw_params)

        if self.match_count == 0:
            if use_mean_filter:
                output_filename = f"{data_dir}_first_match_extractor:{self.feature_extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}_std:{discard_std_multiplier}.png"
            else:
                output_filename = f"{data_dir}_first_match_extractor:{self.feature_extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}.png"
            cv2.imwrite(output_filename, img3)
            print(f"Saved first match image for {self.feature_extractor} as {output_filename}")

        if self.show:
            plt.figure(1); plt.clf()
            plt.imshow(img3)
            plt.title('Number')
            plt.pause(0.1)

        self.match_count += 1


        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([self.kp2[m.trainIdx].pt for m in good])

        # Calculate Euclidean distances for all matches
        distances = [np.linalg.norm(pt1 - pt2) for pt1, pt2 in zip(q1, q2)]

        if use_mean_filter and len(good) > 16:
            # Calculate mean and standard deviation of the distances
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)

            # Set the upper threshold for filtering matches
            upper_threshold = mean_distance + discard_std_multiplier * std_distance

            # Filter matches based on the threshold
            filtered_q1 = []
            filtered_q2 = []
            filtered_distances = []
            for pt1, pt2, dist in zip(q1, q2, distances):
                if dist <= upper_threshold:
                    filtered_q1.append(pt1)
                    filtered_q2.append(pt2)
                    filtered_distances.append(dist)

            q1 = np.float32(filtered_q1)
            q2 = np.float32(filtered_q2)
            distances = filtered_distances

        return q1, q2, distances



    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        if E is None:
            return None

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        if np.isnan(t).any():
            t = np.array([0.0, 0.0, 0.0])

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))

        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        R1, R2, t = cv2.decomposeEssentialMat(E)

        # Decompose the essential matrix
        #R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def main():
    # cv2.destroyAllWindows()
    data_dir = "KITTI_sequence_2"  # Try KITTI_sequence_2 too
    #data_dir = "TUM_sequence"
    feature_extractors = ['orb', 'mobilenet_v2', 'vgg']
    #feature_extractors = ['orb']
    max_features = [3000, 6000]
    #max_features = [3000]
    use_mean_filters = [False, True]
    #use_mean_filters = [False]
    std_multipliers = [1, 2]
    #std_multipliers = [4]

    for extractor in feature_extractors:
        for nfeatures in max_features:
            for use_mean_filter in use_mean_filters:
                if use_mean_filter:
                    for std_multiplier in std_multipliers:
                        run_visual_odometry(extractor, nfeatures, use_mean_filter, std_multiplier, data_dir)
                else:
                    std_multiplier = None
                    run_visual_odometry(extractor, nfeatures, use_mean_filter, std_multiplier, data_dir)

def write_to_csv(extractor, nfeatures, use_mean_filter, std_multiplier, final_error, avg_processing_time, data_dir):
    csv_filename = "{}_visual_odometry_errors.csv".format(data_dir)
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['extractor', 'nfeatures', 'use_mean_filter', 'std_multiplier', 'final_error', 'avg_processing_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'extractor': extractor,
            'nfeatures': nfeatures,
            'use_mean_filter': use_mean_filter,
            'std_multiplier': std_multiplier,
            'final_error': final_error,
            'avg_processing_time': avg_processing_time
        })

def plot_bar_plot(data_dir, distances_list, n_bins=1000, test_params=None):
    # Combine all distances from different frames
    all_distances = np.hstack(distances_list)

    # Discretize distances into bins and count occurrences
    counts, bin_edges = np.histogram(all_distances, bins=n_bins)

    # Calculate mean and standard deviation
    mean_distance = np.mean(all_distances)
    std_distance = np.std(all_distances)

    # Plot the bar plot
    plt.figure()
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges))
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.title("Discretized Bar Plot of Euclidean Pixel Distances")

    # Add mean and standard deviation lines
    ylim = plt.ylim()
    plt.axvline(mean_distance, color='r', linestyle='-', label=f'Mean: {mean_distance:.2f}')
    plt.axvline(mean_distance - std_distance, color='g', linestyle='--', label=f'-1 Std: {mean_distance - std_distance:.2f}')
    plt.axvline(mean_distance + std_distance, color='g', linestyle='--', label=f'+1 Std: {mean_distance + std_distance:.2f}')
    plt.legend(loc='upper right')


    # Add test parameters as a text annotation with white background
    if test_params is not None:
        param_str = '\n'.join([f'{k}={v}' for k, v in test_params.items()])
        plt.annotate(f'Test Parameters:\n{param_str}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))

    # Save the plot to a file in the same folder as the script with parameters in the filename
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if test_params is not None:
        param_str = '_'.join([f'{k}-{v}' for k, v in test_params.items()])
        filename = f'{data_dir}_bar_plot_{param_str}.png'
    else:
        filename = '{data_dir}_bar_plot.png'
    output_path = os.path.join(script_dir, filename)
    plt.savefig(output_path)
    plt.close()

def run_visual_odometry(extractor, nfeatures, use_mean_filter, std_multiplier, data_dir):
    print(f"Running visual odometry with {extractor} feature extractor, {nfeatures} max features, mean filter {use_mean_filter}, and {std_multiplier} standard deviations...")

    vo = VisualOdometry(data_dir, False, extractor, nfeatures)

    gt_path = []
    estimated_path = []

    all_distances_list = []
    processing_times = []

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        start_time = time.time()
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2, distances = vo.get_matches(data_dir, i, use_mean_filter, std_multiplier, nfeatures)
            
            # Only append the distances if they are returned
            if distances is not None:
                all_distances_list.append(distances)

            transf = vo.get_pose(q1, q2)

            if transf is None:
                cur_pose = cur_pose
            else:
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

            print("image {}/{}".format(i, len(vo.gt_poses)))

        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        end_time = time.time()
        processing_times.append(end_time - start_time)

    avg_processing_time = sum(processing_times) / len(processing_times)

    gt_final_x, gt_final_y = gt_path[-1]
    est_final_x, est_final_y = estimated_path[-1]
    final_error = ((gt_final_x - est_final_x) ** 2 + (gt_final_y - est_final_y) ** 2) ** 0.5

    write_to_csv(extractor, nfeatures, use_mean_filter, std_multiplier, final_error, avg_processing_time, data_dir)

    if use_mean_filter:
        output_path = os.path.join(os.getcwd(), f"{data_dir}_Visual_Odometry_extractor:{extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}_std:{std_multiplier}.png")
        plotting_mpl.visualize_paths(gt_path, estimated_path, output_path, title=f"{data_dir}Visual_Odometry_extractor:{extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}_std:{std_multiplier}", show=vo.show)
    else:
        output_path = os.path.join(os.getcwd(), f"{data_dir}_Visual_Odometry_extractor:{extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}.png")
        plotting_mpl.visualize_paths(gt_path, estimated_path, output_path, title=f"{data_dir}Visual_Odometry_extractor:{extractor}_max_features:{nfeatures}_mean_filter:{use_mean_filter}", show=vo.show)

    test_params = {
        'extractor': extractor,
        'nfeatures': nfeatures,
        'use_mean_filter': use_mean_filter,
        'std_multiplier': std_multiplier
    }
    plot_bar_plot(data_dir, all_distances_list, n_bins=100, test_params=test_params)


if __name__ == "__main__":
    main()