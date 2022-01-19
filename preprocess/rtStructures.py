"""
RTStructures class

Author: Han
"""
import glob

from dicompylercore.dicomparser import DicomParser
from preConfig import roi_list


class RTStructures:
    """
    Read the rtStructures dicom of one patient.
    """

    def __init__(self, path):
        """
        Initialize some attributes.
        :param path: the folder of the patient to be read
        """
        self.path = path
        self.data = DicomParser(sorted(glob.glob(self.path + '/RS*'))[0])
        self.planes = self.get_planes()

    def get_planes(self):
        """
        :return:
            planes: i.e. the rois extracted from the structures of data
        """
        structures = self.data.GetStructures()
        planes = []
        for k, v in structures.items():
            if v['name'] in roi_list:
                planes.append({'id': k, 'name': v['name'], 'data': self.data.GetStructureCoordinates(k)})
        return planes
