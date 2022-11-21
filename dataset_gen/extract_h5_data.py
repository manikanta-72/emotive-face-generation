import h5py

if __name__ == "__main__":
    file_name = "/home/mani/git/emotive-face-generation/data/ck_h5_data/ck_data.h5"
    f = h5py.File(file_name, "r")

#     for key in f.keys():
#         print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
#         print(type(f[key])) # get the object type: usually group or dataset
    print(list(f.keys()))
    print(len(f['data_pixel']))
    print(f['data_pixel'][0].shape)
