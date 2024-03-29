# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/workspace/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/workspace/opencv/build

# Include any dependencies generated for this target.
include CMakeFiles/facedetect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/facedetect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/facedetect.dir/flags.make

CMakeFiles/facedetect.dir/FaceDetect.cpp.o: CMakeFiles/facedetect.dir/flags.make
CMakeFiles/facedetect.dir/FaceDetect.cpp.o: ../FaceDetect.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/workspace/opencv/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/facedetect.dir/FaceDetect.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/facedetect.dir/FaceDetect.cpp.o -c /home/kevin/workspace/opencv/FaceDetect.cpp

CMakeFiles/facedetect.dir/FaceDetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/facedetect.dir/FaceDetect.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/workspace/opencv/FaceDetect.cpp > CMakeFiles/facedetect.dir/FaceDetect.cpp.i

CMakeFiles/facedetect.dir/FaceDetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/facedetect.dir/FaceDetect.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/workspace/opencv/FaceDetect.cpp -o CMakeFiles/facedetect.dir/FaceDetect.cpp.s

CMakeFiles/facedetect.dir/FaceDetect.cpp.o.requires:
.PHONY : CMakeFiles/facedetect.dir/FaceDetect.cpp.o.requires

CMakeFiles/facedetect.dir/FaceDetect.cpp.o.provides: CMakeFiles/facedetect.dir/FaceDetect.cpp.o.requires
	$(MAKE) -f CMakeFiles/facedetect.dir/build.make CMakeFiles/facedetect.dir/FaceDetect.cpp.o.provides.build
.PHONY : CMakeFiles/facedetect.dir/FaceDetect.cpp.o.provides

CMakeFiles/facedetect.dir/FaceDetect.cpp.o.provides.build: CMakeFiles/facedetect.dir/FaceDetect.cpp.o

# Object files for target facedetect
facedetect_OBJECTS = \
"CMakeFiles/facedetect.dir/FaceDetect.cpp.o"

# External object files for target facedetect
facedetect_EXTERNAL_OBJECTS =

facedetect: CMakeFiles/facedetect.dir/FaceDetect.cpp.o
facedetect: /usr/local/lib/libopencv_calib3d.so
facedetect: /usr/local/lib/libopencv_contrib.so
facedetect: /usr/local/lib/libopencv_core.so
facedetect: /usr/local/lib/libopencv_features2d.so
facedetect: /usr/local/lib/libopencv_flann.so
facedetect: /usr/local/lib/libopencv_gpu.so
facedetect: /usr/local/lib/libopencv_highgui.so
facedetect: /usr/local/lib/libopencv_imgproc.so
facedetect: /usr/local/lib/libopencv_legacy.so
facedetect: /usr/local/lib/libopencv_ml.so
facedetect: /usr/local/lib/libopencv_nonfree.so
facedetect: /usr/local/lib/libopencv_objdetect.so
facedetect: /usr/local/lib/libopencv_photo.so
facedetect: /usr/local/lib/libopencv_stitching.so
facedetect: /usr/local/lib/libopencv_ts.so
facedetect: /usr/local/lib/libopencv_video.so
facedetect: /usr/local/lib/libopencv_videostab.so
facedetect: CMakeFiles/facedetect.dir/build.make
facedetect: CMakeFiles/facedetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable facedetect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/facedetect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/facedetect.dir/build: facedetect
.PHONY : CMakeFiles/facedetect.dir/build

CMakeFiles/facedetect.dir/requires: CMakeFiles/facedetect.dir/FaceDetect.cpp.o.requires
.PHONY : CMakeFiles/facedetect.dir/requires

CMakeFiles/facedetect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/facedetect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/facedetect.dir/clean

CMakeFiles/facedetect.dir/depend:
	cd /home/kevin/workspace/opencv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/workspace/opencv /home/kevin/workspace/opencv /home/kevin/workspace/opencv/build /home/kevin/workspace/opencv/build /home/kevin/workspace/opencv/build/CMakeFiles/facedetect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/facedetect.dir/depend

