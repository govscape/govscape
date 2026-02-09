import os
import struct

import numpy as np


class NpyToBin:
    # pass in bin file
    def __init__(self, bin_file, page_indices):
        self.bin_file = bin_file
        self.page_indices = page_indices

    # append a pdf directory of npy files to bin
    def convert_pdfdir_to_bin(self, embedding_directory):
        bin_path = self.bin_file
        page_indices_path = self.page_indices
        file_exists = os.path.exists(bin_path)
        if not file_exists:
            with open(bin_path, "w+b") as file:
                file.write(struct.pack("i", 0))
                file.write(struct.pack("i", 0))

        with (
            open(bin_path, "r+b") as file,
            open(page_indices_path, "a+b") as index_file,
        ):
            file.seek(0)
            total_points = struct.unpack("i", file.read(4))[0]
            dimension = struct.unpack("i", file.read(4))[0]
            file.seek(0, os.SEEK_END)

            for page in os.listdir(embedding_directory):
                if page.endswith(".npy"):
                    npy_file = os.path.join(embedding_directory, page)
                    data = np.load(npy_file, mmap_mode="r")
                    data = np.asarray(data)
                    data = data / np.linalg.norm(data, axis=1, keepdims=True)
                    data_points, data_dimension = data.shape

                    file.write(data.tobytes())
                    total_points += data_points

                    # 114 bytes for pdf name, 4 bytes for page number
                    index_file.write(page[0:113].encode("utf-8"))
                    # if img file take out '_img'
                    if page.endswith("_img", len(page) - 8, len(page) - 4):
                        index_file.write(
                            struct.pack("i", int(page[114 : (len(page) - 8)]))
                        )
                    else:
                        index_file.write(
                            struct.pack("i", int(page[114 : (len(page) - 4)]))
                        )

                    if dimension == 0:
                        dimension = data_dimension
                    elif dimension != data_dimension:
                        raise ValueError(
                            "dimension of vector in file does not match "
                            "dimension of data"
                        )
            file.seek(0)
            file.write(struct.pack("i", total_points))
            file.write(struct.pack("i", dimension))
