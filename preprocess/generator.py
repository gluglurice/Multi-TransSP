"""
Generator class

Author: Han
"""
import os

import SimpleITK as sitk
import cv2
import numpy as np

from origin import Origin
from rtStructures import RTStructures


class Generator:
    """
    Generate the mha of origin and rtStructures.
    Every patient will have one generator.
    """

    def __init__(self, input_path, output_path):
        """
        Initialize some attributes.
        :param input_path: the folder of the patient to be read
        :param output_path: the folder to which the mha(s) will be generated
        """
        self.input_path = input_path
        self.output_path = output_path
        self.package_name = input_path.split('/')[-1]

        self.origin = Origin(input_path)
        self.rtStructures = RTStructures(input_path)

    def generate_mha(self, array, generating_type=''):
        """
        Transform the RTStructures into an mha according to the PTV or OAR type.
        :param: array: numpy array to be transformed to mha
        :param: generating_type: the suffix of the name of the mha to be generated
        :return: None
        """
        if len(generating_type) > 0:
            generating_type = "-" + generating_type
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        pixelspacing = (float(self.origin.pixelspacing[0]),
                        float(self.origin.pixelspacing[1]), float(self.origin.thickness))

        image3D = sitk.GetImageFromArray(arr=array)
        image3D.SetSpacing(pixelspacing)
        image3D.SetOrigin(self.origin.position)
        sitk.WriteImage(image3D, f'{self.output_path}/{self.package_name + generating_type}.mha')

    def generate_origin_mha(self):
        """
        Transform the CT sequence into an mha.
        :return: None
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.input_path)
        reader.SetFileNames(dicom_names)
        image3D = reader.Execute()

        # Output the ".mha" files to "output_path".
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        sitk.WriteImage(image3D, f'{self.output_path}/{self.package_name}.mha')

    def get_rs(self):
        """
        Extract the RTStructures into a numpy array.
        :return:
            rtStructure: the segmentation graph containing all the rois
        """
        rois = [roi['name'] for roi in self.rtStructures.planes] if len(self.rtStructures.planes) > 0 else []
        prone = True if 'p' in self.origin.patient_position.lower() else False
        feet_first = True if 'ff' in self.origin.patient_position.lower() else False
        z = int(self.origin.position[2])

        """Aggregate rtStructures info of all the rois in "roi_list"
        into one numpy array."""
        rtStructure = np.zeros((self.origin.length, self.origin.rows, self.origin.columns), dtype='uint8')
        for roi in rois:
            new = {}
            current_plane = None
            for plane in self.rtStructures.planes:
                if plane['name'] == roi:
                    current_plane = plane
                    break
            for k, v in current_plane['data'].items():
                for item in v:
                    item_z = int(item['data'][0][2])
                    item['data'] = self.get_contour_pixel_data(self.origin.patient_to_pixel_lut,
                                                               item['data'], prone, feet_first)
                    if new.get(item_z):
                        new[item_z].append(item['data'])
                    else:
                        new[item_z] = [item['data']]
            for k, v in new.items():
                res = np.zeros((self.origin.rows, self.origin.columns), dtype='uint8')
                for c in v:
                    # Draw the ROI with white color
                    cv2.fillPoly(res, [np.array(c)], (1, 1, 1))
                rtStructure[round(len(self.origin.origin_list) - 1 - (z - k) / 3)] += res

        for i, x in np.ndenumerate(rtStructure):
            if x != 0:
                rtStructure[i] = 1

        return rtStructure

    def get_mha(self, mha_path, reading_type=''):
        """
        Transform the generated mha(s) into a numpy array.
        :param: mha_path: the path of the mha to be read
        :param: reading_type: the suffix of the name of the mha to be read
        :return:
            image_array: the segmented graph containing all the rois
        """
        if len(reading_type) > 0:
            reading_type = '-' + reading_type
        reading_type += '.mha'

        mha = mha_path + '/' + self.package_name + reading_type
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(mha)).squeeze()

        return image_array

    def get_rs_from_mha(self, mha_path, reading_type=''):
        """
        Extract the RTStructures into a numpy array from generated mha(s).
        :param: mha_path: the path of the mha to be read
        :param: reading_type: the suffix of the name of the mha to be read
        :return:
            rtStructure: the segmentation graph containing all the rois
        """
        rtStructure = self.get_mha(mha_path, reading_type)
        rtStructure = (rtStructure - np.min(rtStructure)) / (np.max(rtStructure) - np.min(rtStructure))
        for i, x in np.ndenumerate(rtStructure):
            if x != 0:
                rtStructure[i] = 1
        return rtStructure.astype('uint8')

    def generate_rs_and_origin_mha(self, rtStructure):
        """
        Multiply the origin by rs, and generate the result mha.
        :param: rtStructure: the rtStructure numpy array
        :return: None
        """
        rs_and_origin_array = rtStructure.astype('int16')

        for i, origin_dcm in enumerate(self.origin.origin_list):
            origin_image_array = sitk.GetArrayFromImage(sitk.ReadImage(origin_dcm)).squeeze()
            rs_and_origin_array[i] *= origin_image_array
            for j, x in np.ndenumerate(rs_and_origin_array[i]):
                if x == 0:
                    rs_and_origin_array[i][j] = -1024

        self.generate_mha(rs_and_origin_array, 'rs+origin')

    @staticmethod
    def get_min_max_layer(rtStructure):
        """
        Get the number tuples of min and max layers.
        :param: rtStructure: the rtStructure numpy array
        :return:
            min_layer: number of the min layer which contains rs info.
            max_layer: number of the max layer which contains rs info.
        """
        min_layer, max_layer = 0, 0
        for i, layer in enumerate(rtStructure):
            if np.any(layer):
                if min_layer == 0:
                    min_layer = i + 1
                max_layer = i + 1
        return min_layer, max_layer

    @staticmethod
    def get_contour_pixel_data(patient_to_pixel_lut, contour, prone=False, feetfirst=False):
        """
        Convert structure data into pixel data using the patient-to-pixel-LUT.
        :param patient_to_pixel_lut:
        :param contour:
        :param prone:
        :param feetfirst:
        :return:
            pixel_data: contour pixel data
        """
        pixel_data = []
        """For each point in the structure data
        look up the value in the LUT and find the corresponding pixel pair"""
        for point in contour:
            x, y = None, None
            for xv, xval in enumerate(patient_to_pixel_lut[0]):
                if xval > point[0] and not prone and not feetfirst:
                    x = xv
                    break
                elif xval < point[0] and (feetfirst or prone):
                    x = xv
                    break
            for yv, yval in enumerate(patient_to_pixel_lut[1]):
                if yval > point[1] and not prone:
                    y = yv
                    break
                elif yval < point[1] and prone:
                    y = yv
                    break
            if x is not None and y is not None:
                pixel_data.append((x, y))

        return pixel_data
