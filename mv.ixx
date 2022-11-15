module;

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <list>
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
#include <msgpack.hpp>
#include <opencv2/core.hpp>

#include <CameraParams.h>
#include <MvErrorDefine.h>
#include <MvCameraControl.h>

#define _MVS_ENUM_EXPAND(OP, OS, ON, EP, ES, EN) EP##EN##ES = MV_##OP##ON##OS

#define _MVS_DEVICE_TYPE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(, _DEVICE, ON, , , EN)
#define _MVS_DEVICE_TYPE_SIMPLE_EXPAND(N) _MVS_DEVICE_TYPE_EXPAND(N, N)

#define _MVS_ACCESS_MODE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(ACCESS_, , ON, , , EN)

#define _MVS_TRIGGER_MODE_EXPAND(N) _MVS_ENUM_EXPAND(TRIGGER_MODE_, , N, , , N)

#define _MVS_TRIGGER_SOURCE_EXPAND(ON, EN) _MVS_ENUM_EXPAND(TRIGGER_SOURCE_, , ON, , , EN)

#define _MVS_SIMPLE_REPLACE_CALL(FUNCTION, NAME) \
	[[nodiscard]] \
	bool NAME() & noexcept \
	{ \
		return _wrap(MV_CC_##FUNCTION, _handle); \
	}

#define _MVS_VALUE_SET_CALL(FUNCTION, TYPE, DEST_TYPE, NAME, VALUE_STRING, CAST, ...) \
	[[nodiscard]] \
	bool NAME(TYPE value) & noexcept \
	{ \
		return _wrap( \
			MV_CC_Set##FUNCTION, \
			_handle, \
			#VALUE_STRING, \
			CAST<DEST_TYPE>(value __VA_OPT__(,) __VA_ARGS__) \
		); \
	}

#define _MVS_ENUM_SET(TYPE, VALUE_STRING) _MVS_VALUE_SET_CALL( \
	EnumValue, \
	TYPE, \
	uint32_t, \
	set, \
	VALUE_STRING, \
	static_cast \
)

#define _MVS_FLOATING_SET(NAME, ...) \
	_MVS_VALUE_SET_CALL(FloatValue, std::floating_point auto, float, set_##NAME, __VA_ARGS__)

#define _MVS_INTEGER_SET(NAME, ...) \
	_MVS_VALUE_SET_CALL(IntValueEx, std::integral auto, int64_t, set_##NAME, __VA_ARGS__)

export module mv;

export namespace mv
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
	bool open(access_mode mode = access_mode::EXCLUSIVE, uint16_t switch_over_key = 0) & noexcept
	{
		return _wrap(MV_CC_OpenDevice, _handle, static_cast<uint32_t>(mode), switch_over_key);
	}

	_MVS_SIMPLE_REPLACE_CALL(CloseDevice, close)

	_MVS_SIMPLE_REPLACE_CALL(StartGrabbing, start)

	_MVS_SIMPLE_REPLACE_CALL(StopGrabbing, stop)

	_MVS_ENUM_SET(trigger_mode, TriggerMode)

	_MVS_ENUM_SET(trigger_source, TriggerSource)

	_MVS_ENUM_SET(trigger_activation, TriggerActivation)

	_MVS_FLOATING_SET(fps, AcquisitionFrameRate, std::clamp, 0.09, 100000)

	_MVS_INTEGER_SET(line_debouncer, LineDebouncerTime, std::clamp, 0, 1000000)

	template<typename Rep, typename Period>
	[[nodiscard]]
	bool set_line_debouncer(const std::chrono::duration<Rep, Period>& delay) & noexcept
	{
		int64_t value;
		if constexpr (not std::is_same_v<decltype(delay), std::chrono::microseconds>)
			value = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
		else
			value = delay.count();

		return set_line_debouncer(value);
	}

	[[nodiscard]]
	bool set_receiver(queue& queue) & noexcept
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

}
