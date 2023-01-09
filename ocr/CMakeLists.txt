if (MSVC)
	create_msvc_cxx_module_target(ocr
		COMPILE_DEFINITIONS
			MSGPACK_USE_DEFINE_MAP
		SYSTEM_INCLUDE_DIRECTORIES
			${ONNXRUNTIME_INCLUDE_DIR}
			${OPENCV_INCLUDE_DIRS}
		LINK_LIBRARIES
			Clipper2
			fmt::fmt
			mio::mio
			msgpackc-cxx
			onnxruntime_providers
			onnxruntime_session
			opencv_core
			opencv_imgproc
	)
	add_dependencies(ocr onnxruntime_providers_cuda)
	add_library(ocr::ocr ALIAS ocr)
else ()
	message(FATAL_ERROR "OCR requires C++20 module support")
endif ()

add_executable(ocr_driver EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(ocr_driver
	PRIVATE
		ocr::ocr
)
