module;

#include <cstddef>
#include <cstdint>

#include <chrono>
#include <exception>
#include <list>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <CameraParams.h>
#include <MvErrorDefine.h>
#include <MvCameraControl.h>

export module mv;

namespace mv
{

export
struct queue final
{
	struct frame final
	{
		uint64_t device_timestamp, host_timestamp;
		cv::Mat content;

		frame(uint64_t device_timestamp, uint64_t host_timestamp, cv::Mat content) noexcept
			: device_timestamp(device_timestamp), host_timestamp(host_timestamp), content(std::move(content)) {}

		~frame() noexcept = default;

		frame(const frame&) = delete;

		frame(frame&&) noexcept = default;

		frame& operator=(const frame&) = delete;

		frame& operator=(frame&&) noexcept = default;
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
		self->_queue.emplace_back(device_timestamp, info->nHostTimeStamp, std::move(image));
	}

	std::mutex _lock;
	uint8_t _rotation;
	std::list<frame> _queue;
public:
	queue(uint8_t rotation) : _lock(), _rotation(rotation % 4), _queue() {}

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

	[[nodiscard]]
	bool get(frame& frame) &
	{
		std::lock_guard guard(_lock);

		if (_queue.empty())
			return false;

		frame = std::move(_queue.front());
		_queue.pop_front();
		return true;
	}
};

export
class device final
{
#define _MVS_DEFINE_ENUM(NAME) enum class NAME : unsigned int

#define _MVS_ENUM_EXPAND(OP, OS, ON, EP, ES, EN) EP##EN##ES = MV_##OP##ON##OS

#define _MVS_DEVICE_TYPE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(, _DEVICE, ON, , , EN)
#define _MVS_DEVICE_TYPE_SIMPLE_EXPAND(N) _MVS_DEVICE_TYPE_EXPAND(N, N)
public:
	_MVS_DEFINE_ENUM(type)
	{
		_MVS_DEVICE_TYPE_EXPAND(UNKNOW, UNKNOWN),
		_MVS_DEVICE_TYPE_EXPAND(GIGE, GIG_E),
		_MVS_DEVICE_TYPE_EXPAND(1394, IEEE_1394),
		_MVS_DEVICE_TYPE_SIMPLE_EXPAND(USB),
		_MVS_DEVICE_TYPE_EXPAND(CAMERALINK, CAMERA_LINK),
		_MVS_DEVICE_TYPE_EXPAND(VIR_GIGE, VIRTUAL_GIG_E),
		_MVS_DEVICE_TYPE_EXPAND(VIR_USB, VIRTUAL_USB),
		_MVS_DEVICE_TYPE_EXPAND(GENTL_GIGE, GEN_T_L_GIG_E)
	};
private:
#define _MVS_ACCESS_MODE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(ACCESS_, , ON, , , EN)
public:
	_MVS_DEFINE_ENUM(access_mode)
	{
		_MVS_ACCESS_MODE_EXPAND(Exclusive, EXCLUSIVE),
		_MVS_ACCESS_MODE_EXPAND(ExclusiveWithSwitch, EXCLUSIVE_WITH_SWITCH),
		_MVS_ACCESS_MODE_EXPAND(Control, CONTROL),
		_MVS_ACCESS_MODE_EXPAND(ControlWithSwitch, CONTROL_WITH_SWITCH),
		_MVS_ACCESS_MODE_EXPAND(ControlSwitchEnable, CONTROL_SWITCH_ENABLE),
		_MVS_ACCESS_MODE_EXPAND(ControlSwitchEnableWithKey, CONTROL_SWITCH_ENABLE_WITH_KEY),
		_MVS_ACCESS_MODE_EXPAND(Monitor, MONITOR)
	};
private:
#define _MVS_EXPOSURE_AUTO_EXPAND(N) _MVS_ENUM_EXPAND(EXPOSURE_AUTO_MODE_, , N, , , N)
public:
	_MVS_DEFINE_ENUM(exposure_auto)
	{
		_MVS_EXPOSURE_AUTO_EXPAND(OFF),
		_MVS_EXPOSURE_AUTO_EXPAND(ONCE),
		_MVS_EXPOSURE_AUTO_EXPAND(CONTINUOUS)
	};
private:
#define _MVS_TRIGGER_MODE_EXPAND(N) _MVS_ENUM_EXPAND(TRIGGER_MODE_, , N, , , N)
public:
	_MVS_DEFINE_ENUM(trigger_mode)
	{
		_MVS_TRIGGER_MODE_EXPAND(OFF),
		_MVS_TRIGGER_MODE_EXPAND(ON)
	};
private:
#define _MVS_TRIGGER_SOURCE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(TRIGGER_SOURCE_, , ON, , , EN)
public:
	_MVS_DEFINE_ENUM(trigger_source)
	{
		_MVS_TRIGGER_SOURCE_EXPAND(LINE0, LINE_0),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE1, LINE_1),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE2, LINE_2),
		_MVS_TRIGGER_SOURCE_EXPAND(LINE3, LINE_3),
		_MVS_TRIGGER_SOURCE_EXPAND(COUNTER0, COUNTER_0),
		_MVS_TRIGGER_SOURCE_EXPAND(SOFTWARE, SOFTWARE),
		_MVS_TRIGGER_SOURCE_EXPAND(FrequencyConverter, FREQUENCY_CONVERTER)
	};
private:
#define _MVS_GRAB_STRATEGY_EXPAND(ON, EN) _MVS_ENUM_EXPAND(GrabStrategy_, , ON, , , EN)
public:
	_MVS_DEFINE_ENUM(grab_strategy)
	{
		_MVS_GRAB_STRATEGY_EXPAND(OneByOne, SEQUENTIAL),
		_MVS_GRAB_STRATEGY_EXPAND(LatestImagesOnly, LATEST),
		_MVS_GRAB_STRATEGY_EXPAND(LatestImages, LATEST_CACHE),
		_MVS_GRAB_STRATEGY_EXPAND(UpcomingImage, NEXT)
	};

	_MVS_DEFINE_ENUM(trigger_activation)
	{
		RISING = 0,
		FALLING = 1
	};
private:
	template<typename F, typename... Args>
		requires std::is_invocable_r_v<int, F, Args...>
	[[nodiscard]]
	static inline bool _wrap(F&& f, Args&&... args)
	{
		return f(std::forward<Args>(args)...) == MV_OK;
	}

	void *_handle;
	std::string _serial;
	type _type;

	device() noexcept : _handle(), _serial(), _type(type::UNKNOWN) {}

	template<typename CF>
	[[nodiscard]]
	inline bool _create(CF&& create, MV_CC_DEVICE_INFO *device_info) noexcept
	{
		if (_handle)
			return false;

		std::string s;
		auto t = static_cast<type>(device_info->nTLayerType);
		switch (t)
		{
			case type::GIG_E:
				s = reinterpret_cast<const char *>(device_info->SpecialInfo.stGigEInfo.chSerialNumber);
				break;
			case type::USB:
				s = reinterpret_cast<const char *>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber);
				break;
			case type::CAMERA_LINK:
				s = reinterpret_cast<const char *>(device_info->SpecialInfo.stCamLInfo.chSerialNumber);
				break;
			default:
				return false;
		}

		if (_wrap(std::forward<CF>(create), &_handle, device_info))
		{
			_serial = std::move(s);
			_type = t;
			return true;
		}
		return false;
	}
public:
	[[nodiscard]]
	static std::vector<device> enumerate(type type) noexcept
	{
		if (
			MV_CC_DEVICE_INFO_LIST devices_info_list;
			_wrap(MV_CC_EnumDevices, static_cast<uint32_t>(type), &devices_info_list)
		)
		{
			std::vector<device> devices;
			if (size_t devices_count = devices_info_list.nDeviceNum)
			{
				devices.reserve(devices_count);
				auto devices_info = devices_info_list.pDeviceInfo;
				for (size_t i = 0; i < devices_count; ++i)
					if (device device; device._create(MV_CC_CreateHandleWithoutLog, devices_info[i]))
						devices.emplace_back(std::move(device));
			}
			return devices;
		}
		return {};
	}

	[[nodiscard]]
	static std::vector<device> enumerate(type type, const std::string& log_path) noexcept
	{
		if (not _wrap(MV_CC_SetSDKLogPath, log_path.c_str()))
			return {};

		MV_CC_DEVICE_INFO_LIST devices_info_list;
		if (
			MV_CC_DEVICE_INFO_LIST devices_info_list;
			_wrap(MV_CC_EnumDevices, static_cast<uint32_t>(type), &devices_info_list)
		)
		{
			std::vector<device> devices;
			if (size_t devices_count = devices_info_list.nDeviceNum)
			{
				devices.reserve(devices_count);
				auto devices_info = devices_info_list.pDeviceInfo;
				for (size_t i = 0; i < devices_count; ++i)
					if (device device; device._create(MV_CC_CreateHandle, devices_info[i]))
						devices.emplace_back(std::move(device));
			}
			return devices;
		}
		return {};
	}

	inline ~device() noexcept
	{
		if (not destroy())
			std::terminate();
	}

	device(const device&) = delete;

	device(device&& other) noexcept : _handle(other._handle), _serial(std::move(other._serial)), _type(other._type)
	{
		other._handle = nullptr;
		other._type = type::UNKNOWN;
	}

	device& operator=(const device&) = delete;

	device& operator=(device&& other) noexcept
	{
		if (not destroy())
			std::terminate();
		_handle = other._handle;
		_serial = std::move(other._serial);
		_type = other._type;
		other._handle = nullptr;
		other._type = type::UNKNOWN;

		return *this;
	}

	[[nodiscard]]
	operator bool() const noexcept
	{
		return _handle;
	}

	[[nodiscard]]
	const std::string& serial() const & noexcept
	{
		return _serial;
	}

	[[nodiscard]]
	std::string serial() && noexcept
	{
		return std::move(_serial);
	}

	[[nodiscard]]
	type type() const noexcept
	{
		return _type;
	}

	[[nodiscard]]
	bool destroy() noexcept
	{
		if (_handle)
		{
			auto ret = _wrap(MV_CC_DestroyHandle, _handle);
			_handle = nullptr;
			_serial.clear();
			_type = type::UNKNOWN;
			return ret;
		}
		return true;
	}

	[[nodiscard]]
	bool open(access_mode mode = access_mode::EXCLUSIVE, unsigned short switch_over_key = 0) noexcept
	{
		return _wrap(MV_CC_OpenDevice, _handle, static_cast<uint32_t>(mode), switch_over_key);
	}
private:
#define _MVS_SIMPLE_REPLACE_CALL(FUNCTION, NAME) \
	[[nodiscard]] \
	bool NAME() noexcept \
	{ \
		return _wrap(MV_CC_##FUNCTION, _handle); \
	}
public:
	_MVS_SIMPLE_REPLACE_CALL(CloseDevice, close)

	_MVS_SIMPLE_REPLACE_CALL(StartGrabbing, start)

	_MVS_SIMPLE_REPLACE_CALL(StopGrabbing, stop)
private:
#define _MVS_SIMPLE_REPLACE_SET_CALL(FUNCTION, NAME, TYPE) \
	[[nodiscard]] \
	bool set_##NAME(TYPE value) noexcept \
	{ \
		return _wrap(MV_CC_Set##FUNCTION, _handle, value); \
	}
public:
	_MVS_SIMPLE_REPLACE_SET_CALL(ImageNodeNum, cache_size, unsigned int)

	_MVS_SIMPLE_REPLACE_SET_CALL(OutputQueueSize, queue_size, unsigned int)
private:
#define _MVS_REPLACE_ENUM_SET_CALL(FUNCTION, NAME, TYPE, OLD_TYPE) \
	[[nodiscard]] \
	bool set_##NAME(TYPE value) noexcept \
	{ \
		return _wrap(MV_CC_Set##FUNCTION, _handle, static_cast<MV_##OLD_TYPE>(value)); \
	}
public:
	_MVS_REPLACE_ENUM_SET_CALL(GrabStrategy, grab_strategy, grab_strategy, GRAB_STRATEGY)
private:
#define _MVS_VALUE_SET_CALL(FUNCTION, TYPE, DEST_TYPE, NAME, VALUE_STRING) \
	[[nodiscard]] \
	bool NAME(TYPE value) noexcept \
	{ \
		return _wrap(MV_CC_Set##FUNCTION, _handle, #VALUE_STRING, static_cast<DEST_TYPE>(value)); \
	}

#define _MVS_ENUM_SET(TYPE, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(EnumValue, TYPE, unsigned int, set, VALUE_STRING)
public:
	_MVS_ENUM_SET(exposure_auto, ExposureAuto)

	_MVS_ENUM_SET(trigger_activation, TriggerActivation)

	_MVS_ENUM_SET(trigger_mode, TriggerMode)

	_MVS_ENUM_SET(trigger_source, TriggerSource)
private:
#define _MVS_FLOATING_SET(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(FloatValue, float, float, set_##NAME, VALUE_STRING)
public:
	_MVS_FLOATING_SET(exposure_time, ExposureTime)

	_MVS_FLOATING_SET(fps, AcquisitionFrameRate)

	_MVS_FLOATING_SET(gain, Gain)
private:
#define _MVS_INTEGER_SET(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_CALL(IntValueEx, int64_t, int64_t, set_##NAME, VALUE_STRING)
public:
	_MVS_INTEGER_SET(line_debouncer, LineDebouncerTime)

	[[nodiscard]]
	inline bool set_line_debouncer(const std::chrono::microseconds& delay) noexcept
	{
		return set_line_debouncer(delay.count());
	}

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool set_line_debouncer(const std::chrono::duration<Rep, Period>& delay) noexcept
	{
		return set_line_debouncer(std::chrono::duration_cast<std::chrono::microseconds>(delay));
	}

	[[nodiscard]]
	bool set_receiver(queue& queue) noexcept
	{
		return _wrap(MV_CC_RegisterImageCallBackForBGR, _handle, queue::_callback, &queue);
	}
};

}
