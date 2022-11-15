module;

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <opencv2/core.hpp>

#include <CameraParams.h>
#include <CNNSingleCharDetectCpp.h>
#include <MVD_ErrorDefine.h>
#include <MvErrorDefine.h>
#include <MvCameraControl.h>

#define _MVS_ENUM_EXPAND(OP, OS, ON, EP, ES, EN) EP##EN##ES = MV_##OP##ON##OS

#define _MVS_DEVICE_TYPE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(, _DEVICE, ON, , , EN)
#define _MVS_DEVICE_TYPE_SIMPLE_EXPAND(N) _MVS_DEVICE_TYPE_EXPAND(N, N)

#define _MVS_ACCESS_MODE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(ACCESS_, , ON, , , EN)

#define _MVS_EXPOSURE_AUTO_EXPAND(N) _MVS_ENUM_EXPAND(EXPOSURE_AUTO_MODE_, , N, , , N)

#define _MVS_TRIGGER_MODE_EXPAND(N) _MVS_ENUM_EXPAND(TRIGGER_MODE_, , N, , , N)

#define _MVS_TRIGGER_SOURCE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(TRIGGER_SOURCE_, , ON, , , EN)

#define _MVS_SIMPLE_REPLACE_CALL(FUNCTION, NAME) \
	bool NAME() & noexcept \
	{ \
		return _wrap(MV_CC_##FUNCTION, _handle); \
	}

#define _MVS_VALUE_SET_CALL(FUNCTION, TYPE, DEST_TYPE, NAME, VALUE_STRING) \
	bool NAME(TYPE value) & noexcept \
	{ \
		return _wrap( \
			MV_CC_Set##FUNCTION, \
			_handle, \
			#VALUE_STRING, \
			static_cast<DEST_TYPE>(value) \
		); \
	}

#define _MVS_ENUM_SET(TYPE, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(EnumValue, TYPE, uint32_t, set, VALUE_STRING)

#define _MVS_FLOATING_SET(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(FloatValue, std::floating_point auto, float, set_##NAME, VALUE_STRING)

#define _MVS_INTEGER_SET(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(IntValueEx, std::integral auto, int64_t, set_##NAME, VALUE_STRING)


#define _MVD_PLATFORM_EXPAND(TYPE) TYPE = VisionDesigner::MVD_PLATFORM_TYPE::MVD_PLATFORM_##TYPE

export module mv;

namespace mv
{

namespace sdk
{

namespace
{

template<typename F, typename... Args>
	requires std::is_invocable_r_v<uint32_t, F, Args...>
[[nodiscard]]
inline static bool _wrap(F&& f, Args&&... args)
{
	return f(std::forward<Args>(args)...) == MV_OK;
}

template<typename S, typename F, typename... Args>
	requires std::is_invocable_r_v<uint32_t, F, Args...>
inline static void _raise_if_fail(S&& what, F&& f, Args&&... args)
{
	if (uint32_t ret = f(std::forward<Args>(args)...); ret != MV_OK)
	[[unlikely]]
		throw std::runtime_error(fmt::format(FMT_COMPILE("({:#X}) {}"), ret, what));
}

}

export struct queue final
{
	struct frame final
	{
		struct
		{
			uint64_t device, host;
		} timestamp;

		cv::Mat content;
	};
private:
	friend class device;

	static constexpr cv::RotateFlags _ROTATIONS[] {
		cv::RotateFlags::ROTATE_90_CLOCKWISE,
		cv::RotateFlags::ROTATE_180,
		cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE
	};

	static void _callback(unsigned char *data, MV_FRAME_OUT_INFO_EX *info, void *user) noexcept
	{
		auto self = reinterpret_cast<queue *>(user);

		uint64_t device_timestamp = uint32_t(info->nDevTimeStampHigh);
		device_timestamp <<= 32;
		device_timestamp |= uint32_t(info->nDevTimeStampLow);

		cv::Mat buffer(info->nHeight, info->nWidth, CV_8UC3, data), image;
		if (self->_rotation)
			cv::rotate(buffer, image, _ROTATIONS[self->_rotation - 1]);
		else
			image = buffer.clone();

		std::lock_guard guard(self->_lock);
		self->_queue.push_back({
			.timestamp = {
				.device = device_timestamp,
				.host = uint64_t(info->nHostTimeStamp)
			},
			.content = std::move(image)
		});
	}

	mutable std::mutex _lock;
	uint8_t _rotation;
	std::list<frame> _queue;
public:
	queue(std::unsigned_integral auto rotation) : _lock(), _rotation(rotation % 4), _queue() {}

	queue(const queue&) = delete;

	queue(queue&& other) noexcept : _lock(), _queue()
	{
		std::lock_guard guard(other._lock);
		_rotation = other._rotation;
		_queue = std::move(other._queue);
	}

	queue& operator=(const queue&) = delete;

	queue& operator=(queue&& other) = delete;

	~queue() noexcept = default;

	void clear() &
	{
		std::lock_guard guard(_lock);
		_queue.clear();
	}

	std::optional<frame> get() &
	{
		std::lock_guard guard(_lock);

		if (_queue.empty())
			return std::nullopt;

		auto ret = std::move(_queue.front());
		_queue.pop_front();
		return std::move(ret);
	}
};

export struct device final
{
	enum class type : uint32_t
	{
		_MVS_DEVICE_TYPE_SIMPLE_EXPAND(UNKNOW),
		_MVS_DEVICE_TYPE_EXPAND(GIGE, GIG_E),
		_MVS_DEVICE_TYPE_EXPAND(1394, IEEE_1394),
		_MVS_DEVICE_TYPE_SIMPLE_EXPAND(USB),
		_MVS_DEVICE_TYPE_EXPAND(CAMERALINK, CAMERA_LINK),
		_MVS_DEVICE_TYPE_EXPAND(VIR_GIGE, VIRTUAL_GIG_E),
		_MVS_DEVICE_TYPE_EXPAND(VIR_USB, VIRTUAL_USB),
		_MVS_DEVICE_TYPE_EXPAND(GENTL_GIGE, GEN_T_L_GIG_E)
	};

	enum class access_mode : uint32_t
	{
		_MVS_ACCESS_MODE_EXPAND(Exclusive, EXCLUSIVE),
		_MVS_ACCESS_MODE_EXPAND(ExclusiveWithSwitch, EXCLUSIVE_WITH_SWITCH),
		_MVS_ACCESS_MODE_EXPAND(Control, CONTROL),
		_MVS_ACCESS_MODE_EXPAND(ControlWithSwitch, CONTROL_WITH_SWITCH),
		_MVS_ACCESS_MODE_EXPAND(ControlSwitchEnable, CONTROL_SWITCH_ENABLE),
		_MVS_ACCESS_MODE_EXPAND(ControlSwitchEnableWithKey, CONTROL_SWITCH_ENABLE_WITH_KEY),
		_MVS_ACCESS_MODE_EXPAND(Monitor, MONITOR)
	};

	enum class exposure_auto : uint32_t
	{
		_MVS_EXPOSURE_AUTO_EXPAND(OFF),
		_MVS_EXPOSURE_AUTO_EXPAND(ONCE),
		_MVS_EXPOSURE_AUTO_EXPAND(CONTINUOUS)
	};

	enum class trigger_mode : uint32_t
	{
		_MVS_TRIGGER_MODE_EXPAND(OFF),
		_MVS_TRIGGER_MODE_EXPAND(ON)
	};

	enum class trigger_source : uint32_t
	{
		_MVS_TRIGGER_SOURCE_EXPAND(LINE0, LINE_0),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE1, LINE_1),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE2, LINE_2),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE3, LINE_3),
		_MVS_TRIGGER_SOURCE_EXPAND(COUNTER0, COUNTER_0),
		_MVS_TRIGGER_SOURCE_EXPAND(SOFTWARE, SOFTWARE),
		_MVS_TRIGGER_SOURCE_EXPAND(FrequencyConverter, FREQUENCY_CONVERTER)
	};

	enum class trigger_activation : uint32_t
	{
		RISING = 0,
		FALLING = 1
	};

	[[nodiscard]]
	static std::vector<device> enumerate(
		type type,
		std::optional<std::string> log_path = std::nullopt
	) noexcept
	{
		MV_CC_DEVICE_INFO_LIST devices_info_list;
		if (_wrap(MV_CC_EnumDevices, static_cast<uint32_t>(type), &devices_info_list))
		{
			std::vector<device> devices;
			if (size_t devices_count = devices_info_list.nDeviceNum)
			{
				if (log_path and log_path->size())
					if (_wrap(MV_CC_SetSDKLogPath, log_path->c_str()))
						return {};

				devices.reserve(devices_count);
				auto devices_info = devices_info_list.pDeviceInfo;
				for (size_t i = 0; i < devices_count; ++i)
				{
					auto device_info = devices_info[i];
					device device;
					switch (type)
					{
						case type::GIG_E:
							device._serial = reinterpret_cast<const char *>(
								device_info->SpecialInfo.stGigEInfo.chSerialNumber
							);
							break;
						case type::USB:
							device._serial = reinterpret_cast<const char *>(
								device_info->SpecialInfo.stUsb3VInfo.chSerialNumber
							);
							break;
						case type::CAMERA_LINK:
							device._serial = reinterpret_cast<const char *>(
								device_info->SpecialInfo.stCamLInfo.chSerialNumber
							);
							break;
						default:
							continue;
					}
					if (_wrap(
						log_path ? MV_CC_CreateHandle : MV_CC_CreateHandleWithoutLog,
						&device._handle,
						device_info
					))
						devices.emplace_back(std::move(device));
				}
			}
			return devices;
		}
		return {};
	}
private:
	void *_handle;
	std::string _serial;

	device() noexcept : _handle(), _serial() {}
public:
	~device() noexcept
	{
		if (_handle)
		{
			_raise_if_fail("failed to destroy handle", MV_CC_DestroyHandle, _handle);
			_handle = nullptr;
		}
	}

	device(const device&) = delete;

	device(device&& other) noexcept : _handle(other._handle), _serial(std::move(other._serial))
	{
		other._handle = nullptr;
	}

	device& operator=(const device&) = delete;

	device& operator=(device&& other) = delete;

	operator bool() const & noexcept
	{
		return _handle;
	}

	bool clear() & noexcept
	{
		auto handle = _handle;
		_handle = nullptr;
		return _wrap(MV_CC_DestroyHandle, handle);
	}

	[[nodiscard]]
	bool open(access_mode mode = access_mode::EXCLUSIVE, uint16_t switch_over_key = 0) & noexcept
	{
		return _wrap(MV_CC_OpenDevice, _handle, static_cast<uint32_t>(mode), switch_over_key);
	}

	_MVS_SIMPLE_REPLACE_CALL(CloseDevice, close)

	_MVS_SIMPLE_REPLACE_CALL(StartGrabbing, start)

	_MVS_SIMPLE_REPLACE_CALL(StopGrabbing, stop)

	_MVS_ENUM_SET(exposure_auto, ExposureAuto)

	_MVS_ENUM_SET(trigger_mode, TriggerMode)

	_MVS_ENUM_SET(trigger_source, TriggerSource)

	_MVS_ENUM_SET(trigger_activation, TriggerActivation)

	_MVS_FLOATING_SET(exposure_time, ExposureTime)

	_MVS_FLOATING_SET(fps, AcquisitionFrameRate)

	_MVS_FLOATING_SET(gain, Gain)

	_MVS_INTEGER_SET(line_debouncer, LineDebouncerTime)

	template<typename Rep, typename Period>
	bool set_line_debouncer(const std::chrono::duration<Rep, Period>& delay) & noexcept
	{
		int64_t value;
		if constexpr (not std::is_same_v<decltype(delay), std::chrono::microseconds>)
			value = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
		else
			value = delay.count();

		return set_line_debouncer(value);
	}

	bool set_receiver(queue& queue) & noexcept
	{
		return _wrap(MV_CC_RegisterImageCallBackForBGR, _handle, queue::_callback, &queue);
	}

	const std::string& serial() const & noexcept
	{
		return _serial;
	}
};

}

namespace detection
{

namespace
{

template<typename F, typename... Args>
	requires std::is_invocable_r_v<int, F, Args...>
[[nodiscard]]
inline static bool _wrap(F&& f, Args&&... args)
{
	return f(std::forward<Args>(args)...) == MVD_OK;
}

template<typename S, typename F, typename... Args>
	requires std::is_invocable_r_v<int, F, Args...>
inline static void _raise_if_fail(S&& what, F&& f, Args&&... args)
{
	if (int ret = f(std::forward<Args>(args)...); ret != MVD_OK)
	[[unlikely]]
		throw std::runtime_error(fmt::format(FMT_COMPILE("({:#X}) {}"), ret, what));
}

}

export struct character final
{
	using result_type = std::vector<std::string>;

	enum class device
	{
		_MVD_PLATFORM_EXPAND(CPU),
		_MVD_PLATFORM_EXPAND(GPU)
	};
private:
	VisionDesigner::CNNSingleCharDetect::ICNNSingleCharDetectTool *_ocr;
	VisionDesigner::IMvdImage *_image;
public:
	character(device device) : _ocr(), _image()
	{
		_raise_if_fail(
			"failed to create detection tool",
			CreateCNNSingleCharDetectToolInstance,
			&_ocr,
			static_cast<VisionDesigner::MVD_PLATFORM_TYPE>(device)
		);
		_raise_if_fail("failed to create image", CreateImageInstance, &_image);
	}

	character(const character&) = delete;

	character(character&& other) noexcept : _ocr(other._ocr), _image(other._image)
	{
		other._ocr = nullptr;
		other._image = nullptr;
	}

	character& operator=(const character&) = delete;

	character& operator=(character&& other) = delete;

	~character() noexcept
	{
		if (_ocr)
		{
			_raise_if_fail(
				"failed to destroy detection tool",
				DestroyCNNSingleCharDetectToolInstance,
				_ocr
			);
			_ocr = nullptr;
		}

		if (_image)
		{
			_raise_if_fail("failed to destroy image", DestroyImageInstance, _image);
			_image = nullptr;
		}
	}

	[[nodiscard]]
	result_type operator()(const cv::Mat& content) &
	{
		_image->InitImage(
			content.cols,
			content.rows,
			VisionDesigner::MVD_PIXEL_FORMAT::MVD_PIXEL_BGR_BGR24_C3
		);
		auto channel_info = _image->GetImageData(0);
		content.copyTo(cv::Mat(
			content.rows,
			content.cols,
			CV_8UC3,
			channel_info->pData,
			channel_info->nRowStep
		));

		_ocr->SetInputImage(_image);
		_ocr->Run();
		auto inferences = _ocr->GetResult();

		std::vector<std::string> ret;
		const size_t lines_count = inferences->GetTextNum();
		ret.reserve(lines_count);
		for (size_t i = 0; i < lines_count; ++i)
		{
			auto line_result = inferences->GetTextInfo(i);
			auto info_list = line_result->GetRecogInfoList();
			auto& line = ret.emplace_back();

			const size_t blocks_count = info_list->GetTopNum();
			for (size_t j = 0; j < blocks_count; ++j)
			{
				auto info = info_list->GetRecogInfo(j);
				line.append(info->GetRecogString(), info->GetCharNum());
			}
		}
		return ret;
	}
};

}

}
