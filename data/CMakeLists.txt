# set property that will initiate a configure step if data directory is modified
set_property(DIRECTORY APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR})

set(COPIED_DATA_FILES)
file(GLOB DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(DATA_FILE ${DATA_FILES})
  get_filename_component(DATA_FILENAME ${DATA_FILE} NAME)
  set(COPIED_DATA_FILE ${CMAKE_CURRENT_BINARY_DIR}/${DATA_FILENAME})
  add_custom_command(OUTPUT ${COPIED_DATA_FILE} DEPENDS ${DATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${DATA_FILE} ${COPIED_DATA_FILE}
    COMMENT "")
  list(APPEND COPIED_DATA_FILES ${COPIED_DATA_FILE})
endforeach()

add_custom_target(copy_data DEPENDS ${COPIED_DATA_FILES}
  COMMENT "Synching the data directory ...")
