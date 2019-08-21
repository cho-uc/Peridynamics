
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(ReadLegacyUnstructuredGrid)

find_package(VTK COMPONENTS 
  vtkChartsCore
  vtkCommonColor
  vtkCommonCore
  vtkCommonDataModel
  vtkFiltersCore
  vtkFiltersExtraction
  vtkFiltersGeneral
  vtkFiltersSources
  vtkIOLegacy
  vtkInteractionStyle
  vtkRenderingContext2D
  vtkRenderingContextOpenGL2
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingGL2PSOpenGL2
  vtkRenderingLabel
  vtkRenderingOpenGL2
  vtkViewsContext2D QUIET)
if (NOT VTK_FOUND)
  message("Skipping ReadLegacyUnstructuredGrid: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  add_executable(ReadLegacyUnstructuredGrid MACOSX_BUNDLE ReadLegacyUnstructuredGrid.cxx )
  target_link_libraries(ReadLegacyUnstructuredGrid PRIVATE ${VTK_LIBRARIES})
else ()
  # include all components
  add_executable(ReadLegacyUnstructuredGrid MACOSX_BUNDLE ReadLegacyUnstructuredGrid.cxx )
  target_link_libraries(ReadLegacyUnstructuredGrid PRIVATE ${VTK_LIBRARIES})
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS ReadLegacyUnstructuredGrid
    MODULES ${VTK_LIBRARIES}
    )
endif () 
