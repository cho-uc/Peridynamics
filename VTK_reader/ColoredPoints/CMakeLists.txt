
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(ColoredPoints)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkCommonCore
  vtkCommonDataModel
  vtkFiltersGeneral
  vtkInteractionStyle
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingOpenGL2 QUIET)
if (NOT VTK_FOUND)
  message("Skipping ColoredPoints: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  add_executable(ColoredPoints MACOSX_BUNDLE ColoredPoints.cxx )
  target_link_libraries(ColoredPoints PRIVATE ${VTK_LIBRARIES})
  target_compile_features(ColoredPoints PUBLIC cxx_std_17)
else ()
  # include all components
  add_executable(ColoredPoints MACOSX_BUNDLE ColoredPoints.cxx )
  target_link_libraries(ColoredPoints PRIVATE ${VTK_LIBRARIES})
  target_compile_features(ColoredPoints PUBLIC cxx_std_17)
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS ColoredPoints
    MODULES ${VTK_LIBRARIES}
    )
endif () 

# add target_compile_features because <algorithm> header can only use --std=c++1z