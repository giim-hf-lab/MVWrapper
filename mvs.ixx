module;

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <future>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <msgpack.hpp>
#include <opencv2/core.hpp>

#include <CameraParams.h>
#include <MvErrorDefine.h>
#include <MvCameraControl.h>

#define _MVS_ENUM_EXPAND(OP, OS, ON, EP, ES, EN) EP##EN##ES = OP##ON##OS

#define _MVS_ERROR_CODE_PREFIX_EXPAND(P, ON, EN) _MVS_ENUM_EXPAND(MV_E_##P, , ON, , , P##EN)
#define _MVS_ERROR_CODE_PREFIX_SIMPLE_EXPAND(P, N) _MVS_ERROR_CODE_PREFIX_EXPAND(P, N, N)

#define _MVS_ERROR_CODE_EXPAND(ON, EN) _MVS_ERROR_CODE_PREFIX_EXPAND(, ON, EN)
#define _MVS_ERROR_CODE_SIMPLE_EXPAND(N) _MVS_ERROR_CODE_EXPAND(N, N)

#define _MVS_GC_ERROR_CODE_EXPAND(ON, EN) _MVS_ERROR_CODE_PREFIX_EXPAND(GC_, ON, EN)
#define _MVS_GC_ERROR_CODE_SIMPLE_EXPAND(N) _MVS_GC_ERROR_CODE_EXPAND(N, N)

#define _MVS_GIG_E_ERROR_CODE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(MV_E_, , ON, GIG_E_, , EN)
#define _MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(N) _MVS_GIG_E_ERROR_CODE_EXPAND(N, N)

#define _MVS_USB_ERROR_CODE_EXPAND(ON, EN) _MVS_ERROR_CODE_PREFIX_EXPAND(USB_, ON, EN)
#define _MVS_USB_ERROR_CODE_SIMPLE_EXPAND(N) _MVS_USB_ERROR_CODE_EXPAND(N, N)

#define _MVS_UPG_ERROR_CODE_EXPAND(ON, EN) _MVS_ERROR_CODE_PREFIX_EXPAND(UPG_, ON, EN)
#define _MVS_UPG_ERROR_CODE_SIMPLE_EXPAND(N) _MVS_UPG_ERROR_CODE_EXPAND(N, N)

#define _MVS_DEVICE_TYPE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(MV_, _DEVICE, ON, , , EN)
#define _MVS_DEVICE_TYPE_SIMPLE_EXPAND(N) _MVS_DEVICE_TYPE_EXPAND(N, N)

#define _MVS_ACCESS_MODE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(MV_ACCESS_, , ON, , , EN)

#define _MVS_TRIGGER_MODE_EXPAND(N) _MVS_ENUM_EXPAND(MV_TRIGGER_MODE_, , N, , , N)

#define _MVS_TRIGGER_SOURCE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(MV_TRIGGER_SOURCE_, , ON, , , EN)

#define _EXPAND_ENUM_SET(T, TS) \
	[[nodiscard]] \
	errc set(T value) & noexcept \
	{ \
		return _wrap(MV_CC_SetEnumValue, _handle, TS, static_cast<uint32_t>(value)); \
	}

export module mvs;

export namespace mvs
{

enum class errc : uint32_t
{
	_MVS_ENUM_EXPAND(MV_, , OK, , , OK),

	_MVS_ERROR_CODE_SIMPLE_EXPAND(HANDLE),
	_MVS_ERROR_CODE_EXPAND(SUPPORT, NOT_SUPPORT),
	_MVS_ERROR_CODE_EXPAND(BUFOVER, BUFFER_OVERFLOW),
	_MVS_ERROR_CODE_EXPAND(CALLORDER, CALL_ORDER),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(PARAMETER),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(RESOURCE),
	_MVS_ERROR_CODE_EXPAND(NODATA, NO_DATA),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(PRECONDITION),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(VERSION),
	_MVS_ERROR_CODE_EXPAND(NOENOUGH_BUF, NOT_ENOUGH_BUFFER),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(ABNORMAL_IMAGE),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(LOAD_LIBRARY),
	_MVS_ERROR_CODE_EXPAND(NOOUTBUF, NO_OUT_BUFFER),
	_MVS_ERROR_CODE_SIMPLE_EXPAND(ENCRYPT),
	_MVS_ERROR_CODE_EXPAND(UNKNOW, UNKNOWN),

	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(GENERIC),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(ARGUMENT),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(RANGE),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(PROPERTY),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(RUNTIME),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(LOGICAL),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(ACCESS),
	_MVS_GC_ERROR_CODE_SIMPLE_EXPAND(TIMEOUT),
	_MVS_GC_ERROR_CODE_EXPAND(DYNAMICCAST, DYNAMIC_CAST),
	_MVS_GC_ERROR_CODE_EXPAND(UNKNOW, UNKNOWN),

	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(NOT_IMPLEMENTED),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(INVALID_ADDRESS),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(WRITE_PROTECT),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(ACCESS_DENIED),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(BUSY),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(PACKET),
	_MVS_GIG_E_ERROR_CODE_EXPAND(NETER, NETWORK_ERROR),
	_MVS_GIG_E_ERROR_CODE_SIMPLE_EXPAND(IP_CONFLICT),

	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(READ),
	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(WRITE),
	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(DEVICE),
	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(GENICAM),
	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(BANDWIDTH),
	_MVS_USB_ERROR_CODE_SIMPLE_EXPAND(DRIVER),
	_MVS_USB_ERROR_CODE_EXPAND(UNKNOW, UNKNOWN),

	_MVS_UPG_ERROR_CODE_SIMPLE_EXPAND(FILE_MISMATCH),
	_MVS_UPG_ERROR_CODE_SIMPLE_EXPAND(LANGUSGE_MISMATCH),
	_MVS_UPG_ERROR_CODE_SIMPLE_EXPAND(CONFLICT),
	_MVS_UPG_ERROR_CODE_SIMPLE_EXPAND(INNER_ERR),
	_MVS_UPG_ERROR_CODE_EXPAND(UNKNOW, UNKNOWN)
};

namespace
{

template<typename F, typename... Args>
	requires std::is_invocable_r_v<uint32_t, F, Args...>
[[nodiscard]]
inline static errc _wrap(F&& f, Args&&... args)
{
	return static_cast<errc>(f(std::forward<Args>(args)...));
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

struct queue final
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

	queue& operator=(queue&& other) noexcept
	{
		std::lock_guard guard1(_lock), guard2(other._lock);
		_rotation = other._rotation;
		_queue = std::move(other._queue);
		return *this;
	}

	~queue() noexcept = default;

	bool available() const & noexcept
	{
		std::unique_lock guard(_lock, std::try_to_lock);
		return guard and _queue.size();
	}

	void clear() &
	{
		std::lock_guard guard(_lock);
		_queue.clear();
	}

	frame get() &
	{
		std::lock_guard guard(_lock);
		auto ret = std::move(_queue.front());
		_queue.pop_front();
		return ret;
	}
};

struct device final
{
	struct camera_setting final
	{
		size_t exposure, gain;

		camera_setting() noexcept = default;

		~camera_setting() noexcept = default;

		camera_setting(const camera_setting&) noexcept = default;

		camera_setting(camera_setting&&) noexcept = default;

		camera_setting& operator=(const camera_setting&) noexcept = default;

		camera_setting& operator=(camera_setting&&) noexcept = default;

		MSGPACK_DEFINE(exposure, gain);
	};

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
	static std::tuple<errc, std::vector<std::tuple<errc, device>>> enumerate(type type) noexcept
	{
		std::vector<std::tuple<errc, device>> maybe_devices;
		MV_CC_DEVICE_INFO_LIST devices_info_list;
		auto ec = _wrap(MV_CC_EnumDevices, static_cast<uint32_t>(type), &devices_info_list);
		if (ec == errc::OK)
		{
			if (size_t devices_count = devices_info_list.nDeviceNum)
			{
				maybe_devices.reserve(devices_count);
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
							maybe_devices.emplace_back(errc::NOT_SUPPORT, std::move(device));
							continue;
					}
					maybe_devices.emplace_back(
						_wrap(MV_CC_CreateHandleWithoutLog, &device._handle, device_info),
						std::move(device)
					);
				}
			}
		}
		return { ec, std::move(maybe_devices) };
	}
private:
	void *_handle;
	std::string _serial;
public:
	device() noexcept : _handle(nullptr), _serial() {}

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

	device& operator=(device&& other) noexcept
	{
		if (_handle)
			_raise_if_fail("failed to destroy handle", MV_CC_DestroyHandle, _handle);

		_handle = other._handle;
		_serial = std::move(other._serial);
		other._handle = nullptr;

		return *this;
	}

	operator bool() const & noexcept
	{
		return _handle;
	}

	[[nodiscard]]
	errc open(access_mode mode = access_mode::EXCLUSIVE, uint16_t switch_over_key = 0) & noexcept
	{
		return _wrap(MV_CC_OpenDevice, _handle, static_cast<uint32_t>(mode), switch_over_key);
	}

	[[nodiscard]]
	errc close() & noexcept
	{
		return _wrap(MV_CC_CloseDevice, _handle);
	}

	[[nodiscard]]
	errc start() & noexcept
	{
		return _wrap(MV_CC_StartGrabbing, _handle);
	}

	[[nodiscard]]
	errc stop() & noexcept
	{
		return _wrap(MV_CC_StopGrabbing, _handle);
	}

	_EXPAND_ENUM_SET(trigger_mode, "TriggerMode")

	_EXPAND_ENUM_SET(trigger_source, "TriggerSource")

	_EXPAND_ENUM_SET(trigger_activation, "TriggerActivation")

	template<typename Rep, typename Period>
	[[nodiscard]]
	errc set_line_debouncer(const std::chrono::duration<Rep, Period>& delay) & noexcept
	{
		int64_t value;
		if constexpr (not std::is_same_v<decltype(delay), std::chrono::microseconds>)
			value = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
		else
			value = delay.count();

		static constexpr int64_t _min = 0, _max = 1000000;
		return _wrap(MV_CC_SetIntValueEx, _handle, "LineDebouncerTime", std::clamp(value, _min, _max));
	}

	[[nodiscard]]
	errc set_receiver(queue& queue) & noexcept
	{
		return _wrap(MV_CC_RegisterImageCallBackForBGR, _handle, queue::_callback, &queue);
	}

	[[nodiscard]]
	const std::string& serial() const & noexcept
	{
		return _serial;
	}
};

}
