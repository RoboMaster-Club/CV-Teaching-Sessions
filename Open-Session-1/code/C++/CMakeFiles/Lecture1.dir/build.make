# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /home/why/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/182.4505.18/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/why/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/182.4505.18/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++

# Include any dependencies generated for this target.
include CMakeFiles/Lecture1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Lecture1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Lecture1.dir/flags.make

CMakeFiles/Lecture1.dir/Lecture1.cpp.o: CMakeFiles/Lecture1.dir/flags.make
CMakeFiles/Lecture1.dir/Lecture1.cpp.o: Lecture1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Lecture1.dir/Lecture1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Lecture1.dir/Lecture1.cpp.o -c /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/Lecture1.cpp

CMakeFiles/Lecture1.dir/Lecture1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Lecture1.dir/Lecture1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/Lecture1.cpp > CMakeFiles/Lecture1.dir/Lecture1.cpp.i

CMakeFiles/Lecture1.dir/Lecture1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Lecture1.dir/Lecture1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/Lecture1.cpp -o CMakeFiles/Lecture1.dir/Lecture1.cpp.s

# Object files for target Lecture1
Lecture1_OBJECTS = \
"CMakeFiles/Lecture1.dir/Lecture1.cpp.o"

# External object files for target Lecture1
Lecture1_EXTERNAL_OBJECTS =

Lecture1: CMakeFiles/Lecture1.dir/Lecture1.cpp.o
Lecture1: CMakeFiles/Lecture1.dir/build.make
Lecture1: /usr/local/lib/libopencv_dnn.so.3.4.3
Lecture1: /usr/local/lib/libopencv_ml.so.3.4.3
Lecture1: /usr/local/lib/libopencv_objdetect.so.3.4.3
Lecture1: /usr/local/lib/libopencv_shape.so.3.4.3
Lecture1: /usr/local/lib/libopencv_stitching.so.3.4.3
Lecture1: /usr/local/lib/libopencv_superres.so.3.4.3
Lecture1: /usr/local/lib/libopencv_videostab.so.3.4.3
Lecture1: /usr/local/lib/libopencv_calib3d.so.3.4.3
Lecture1: /usr/local/lib/libopencv_features2d.so.3.4.3
Lecture1: /usr/local/lib/libopencv_flann.so.3.4.3
Lecture1: /usr/local/lib/libopencv_highgui.so.3.4.3
Lecture1: /usr/local/lib/libopencv_photo.so.3.4.3
Lecture1: /usr/local/lib/libopencv_video.so.3.4.3
Lecture1: /usr/local/lib/libopencv_videoio.so.3.4.3
Lecture1: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
Lecture1: /usr/local/lib/libopencv_imgproc.so.3.4.3
Lecture1: /usr/local/lib/libopencv_core.so.3.4.3
Lecture1: CMakeFiles/Lecture1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Lecture1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Lecture1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Lecture1.dir/build: Lecture1

.PHONY : CMakeFiles/Lecture1.dir/build

CMakeFiles/Lecture1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Lecture1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Lecture1.dir/clean

CMakeFiles/Lecture1.dir/depend:
	cd /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++ && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++ /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++ /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++ /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++ /home/why/CV-Teaching-Sessions/Open-Session-1/code/C++/CMakeFiles/Lecture1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Lecture1.dir/depend
