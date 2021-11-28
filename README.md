# Camera calibration

## Require
- opencv
- dirent

## Make
- `mkdir build && cd build`
- `cmake ..`
- `make`

## Usage
- put chessboard images into `/chessboard`
- put images to be undistorted into `/images`
- put videos to be undistorted into `/videos`
- `./calibrate [-m <chessboard rows>] [-n <chessboard columns>] [-type <normal|fisheye>]`

## Generated files
- intrinsics.yml
   - intrinsic_matrix
   - distortion_coeffs
- undistort.yml
   - map matrix
   - scaled intrinsic_matrix
- corners
   - draw corners images
- undistort
   - undistorted images
   - undistorted videos