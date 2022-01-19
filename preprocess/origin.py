"""
Origin class

Author: Han
"""
import glob

from dicompylercore.dicomparser import DicomParser


class Origin:
    """
    Used to get original CT sequence images of one patient.
    """

    def __init__(self, path):
        """
        Initialize some attributes.
        :param path: the folder of the patient to be read
        """
        self.path = path
        self.origin_list = sorted(glob.glob(self.path + '/CT*'), reverse=False)

        # get some basic info from a sample image
        self.length = len(self.origin_list)
        self.sample = DicomParser(self.origin_list[self.length - 1])

        self.patient_id = self.sample.GetDemographics()['id']
        self.patient_name = self.sample.GetDemographics()['name']
        self.patient_to_pixel_lut = self.sample.GetPatientToPixelLUT()
        self.thickness = self.sample.ds.get('SliceThickness')

        # basic image data of the sample image
        self.image_data = self.sample.GetImageData()

        self.rows = self.image_data['rows']
        self.columns = self.image_data['columns']
        self.orientation = self.image_data['orientation']
        self.pixelspacing = self.image_data['pixelspacing']
        self.patient_position = self.image_data['patientposition']
        self.position = self.image_data['position']
