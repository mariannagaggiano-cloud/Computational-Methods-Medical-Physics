# base script for reading mha files and saving image + array

import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def readIn(fileName):
    reader = sitk.ImageFileReader()

    if os.path.isfile(fileName):
        reader.SetFileName(fileName)
        return reader
    else:
        print("File ", fileName, " not found!")
        sys.exit("Aborting...")


def main():
    print(">>> Executing main ReadMHA <<<\n")

    if len(sys.argv) != 2:
        print("No input file name provided\n")
        sys.exit("Aborting...")

    arg = sys.argv[1]
    print("Input file: ", arg)

    image = readIn(arg)

    # read header
    image.ReadImageInformation()
    print("Image dimensions: ", image.GetDimension())
    print("Image size: ", image.GetSize())
    print("Voxel or pixel size: ", image.GetSpacing())
    print("")

    # read image and convert to numpy
    volume = sitk.GetArrayFromImage(image.Execute())

    # ---- SAVE FULL ARRAY ----
    np.save("volume.npy", volume)
    print("Saved full image array: volume.npy")

    # statistics
    totalsumcounts = np.sum(volume)
    totalareamm = (
        image.GetSize()[0] * image.GetSpacing()[0] *
        image.GetSize()[1] * image.GetSpacing()[1]
    )
    totalcountdensity = totalsumcounts / totalareamm

    print("Total sum of counts: ", totalsumcounts)
    print("Total area: ", totalareamm, " mm2")
    print("Total density: ", totalcountdensity, " counts/mm2")
    print("")

    # ---- SAVE FULL IMAGE ----
    plt.figure()
    plt.title("Shepp-Logan phantom")
    plt.imshow(volume, cmap="gray", vmin=volume.min(), vmax=volume.max())
    plt.xlabel("x in pixels")
    plt.ylabel("y in pixels")
    plt.colorbar(label="Radiological property")
    plt.savefig("counts_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved image: counts_map.png")

    # ROI definition
    #xrange = [190, 210]
    #yrange = [90, 110]

    #volumecropped = volume[yrange[0]:yrange[1], xrange[0]:xrange[1]]

    # ---- SAVE CROPPED ARRAY ----
    #np.save("volume_cropped.npy", volumecropped)
    #print("Saved cropped array: volume_cropped.npy")

    # cropped statistics
    #croppedsumcounts = np.sum(volumecropped)
    #croppedareamm = (
    #    (xrange[1] - xrange[0]) * image.GetSpacing()[0] *
    #    (yrange[1] - yrange[0]) * image.GetSpacing()[1]
    #)
    #croppedcountdensity = croppedsumcounts / croppedareamm

    #print("Cropped sum of counts: ", croppedsumcounts)
    #print("Cropped area: ", croppedareamm, " mm2")
    #print("Cropped density: ", croppedcountdensity, " counts/mm2")

    # ---- SAVE CROPPED IMAGE ----
    #plt.figure()
    #plt.title("Counts map cropped")
    #plt.imshow(volume, cmap="gray", vmin=volume.min(), vmax=volume.max())
    #plt.xlabel("x in pixels")
    #plt.ylabel("y in pixels")
    #plt.colorbar(label="Counts")
    #plt.savefig("counts_map_cropped.png", dpi=300, bbox_inches="tight")
    #plt.close()
    #print("Saved image: counts_map_cropped.png")


if __name__ == "__main__":
    main()