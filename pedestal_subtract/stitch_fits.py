#!/usr/bin/env python3
from astropy.io import fits
import numpy as np
from pathlib import Path
import glob
import argparse

def stitch_fits(file_path, directory='*/', image='avg*.fz', out_path='combined-fits/', print_header=False):
    """
    Stitch FITS files along the y-axis across multiple extensions.

    Parameters:
        file_path (str or Path): Base directory containing files
        directory (str): Subdirectory pattern (glob; accepts wildcard and any bash string logic)
        image (str): File pattern (accepts wildcard and any bash string logic)
        out_path (str or Path): Output directory (can be full or relative path)
        print_header (bool): Whether to print extension headers

    Returns:
        Path: Path to stitched FITS file
    """
    file_path = Path(file_path)

    files = sorted(glob.glob(str(file_path / directory / image), recursive=True))

    # Reformat image name for writing out
    try:
        image_name = '_'.join(n for n in Path(files[0]).name.split('_')[:-3])
    except IndexError:
        print('\nError: no files found matching the specified pattern. Please check your file path, directory, and image patterns.')
        return None

    nfiles = len(files)

    # inspect first file to get dimensions
    ext_headers = []

    # Do not perform stitching if no files found
    if nfiles==0:
        return None
    
    else:
        with fits.open(files[0], memmap=True) as f:
            ny, nx = f[1].data.shape
            nextensions = len(f) - 1

            primary_header = f[0].header.copy()

            for ext in [1,2,3,4]:
                ext_headers.append(f[ext].header.copy())

        if print_header:
            print(ext_headers)

        out_path = file_path / out_path
        out_path.mkdir(parents=True, exist_ok=True)
        outname = out_path / f'{image_name}_{nfiles}_stitched.fits'

        primary_header['NROW'] = primary_header['NROW']*nfiles

        # Create output file
        primary_hdu = fits.PrimaryHDU(header=primary_header)

        hdul = fits.HDUList([primary_hdu])

        for ext in range(1, nextensions + 1):

            big_shape = (ny * nfiles, nx)

            # Preserve compression structure from original files
            hdr = ext_headers[ext-1]

            tile1 = hdr.get("ZTILE1", nx)
            tile2 = hdr.get("ZTILE2", 1)
            cmptype = hdr.get("ZCMPTYPE", "RICE_1")

            hdu = fits.CompImageHDU(
                data=np.zeros(big_shape, dtype=np.float32),
                header=hdr,
                compression_type=cmptype,
                tile_shape=(tile1, tile2)
            )

            # Preserve original extension name if present
            if "EXTNAME" in hdr:
                hdu.name = hdr["EXTNAME"]

            hdu.header["STITCHED"] = nfiles
            hdu.header["SRCFILE"] = files[0]

            hdul.append(hdu)

        hdul.writeto(str(outname), overwrite=True)

        # reopen with memmap
        hdul = fits.open(str(outname), mode="update", memmap=True)

        print("Stitching images...")
        for i, f in enumerate(files):

            with fits.open(f, memmap=True) as infile:

                y0 = i * ny
                y1 = (i + 1) * ny

                for ext in range(1, nextensions + 1):

                    data = infile[ext].data  # (630,20)

                    hdul[ext].data[y0:y1, :] = data

            if i % 5 == 0:
                print(f"{i}/{nfiles}")

        print(f"{nfiles}/{nfiles}")
        hdul.close()
        if outname.exists():
            print(f'successfully saved stitched file to {outname}')
        else:
            print('file not saved correctly')

        return outname


def init_argparse():
    """
    Initializes the ArgumentParser object and defines arguments.
    """
    parser = argparse.ArgumentParser(description="Stitch FITS files across extensions.")

    parser.add_argument("file_path", type=str, help="Base directory containing files")
    parser.add_argument("-d", "--directory", type=str, default="03*/", help="Subdirectory pattern (glob)")
    parser.add_argument("-i", "--image", type=str, default="avg*.fz", help="File pattern")
    parser.add_argument("-o", "--out_path", type=str, default="combined-fits/", help="Output directory")
    parser.add_argument("-p", "--print_header", action="store_true", help="Print extension headers")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = init_argparse()
    
    outname = stitch_fits(
        file_path=args.file_path,
        directory=args.directory,
        image=args.image,
        out_path=args.out_path,
        print_header=args.print_header
    )

    print(f"Output file: {outname}")
    