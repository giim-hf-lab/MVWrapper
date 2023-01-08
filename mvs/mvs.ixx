module;

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <concepts>
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
struct frame final
{
	uint64_t device_timestamp, host_timestamp;
	cv::Mat content;

	frame() noexcept : content() {}

	frame(uint64_t device_timestamp, uint64_t host_timestamp, cv::Mat content) noexcept
		: device_timestamp(device_timestamp), host_timestamp(host_timestamp), content(std::move(content))
	{}

	~frame() noexcept = default;

	frame(const frame&) = delete;

	frame(frame&&) noexcept = default;

	frame& operator=(const frame&) = delete;

	frame& operator=(frame&&) noexcept = default;

	operator bool() const noexcept
	{
		return not content.empty();
	}
};

export
class queue final
{
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
	queue(uint8_t rotation = 0) : _lock(), _rotation(rotation % 4), _queue() {}

	queue(const queue&) = delete;

	queue(queue&& other) noexcept : _lock(), _queue()
	{
		std::lock_guard guard(other._lock);
		_rotation = other._rotation;
		_queue = std::move(other._queue);
	}

	queue& operator=(const queue&) = delete;

	queue& operator=(queue&&) = delete;

	~queue() noexcept = default;

	void clear() &
	{
		std::lock_guard guard(_lock);
		_queue.clear();
	}

	[[nodiscard]]
	bool get(frame& frame, size_t& left) &
	{
		std::lock_guard guard(_lock);

		if (_queue.empty())
			return false;

		frame = std::move(_queue.front());
		_queue.pop_front();
		left = _queue.size();
		return true;
	}
};

export template<typename> struct constraint final {};

export
template<std::integral T>
struct constraint<T> final
{
	T current, min, max, step;

	constexpr constraint() noexcept = default;

	constexpr constraint(T current, T min, T max, T step) noexcept
		: current(current), min(min), max(max), step(step)
	{}

	template<bool round_up>
	constexpr T coerce(T value) const noexcept
	{
		if (value < min)
			return min;
		if (value > max)
			return max;
		if (auto r = (value - min) % step)
			if constexpr (round_up)
				value += step - r;
			else
				value -= r;
		return value;
	}

	inline constexpr T coerce(T value, bool round_up = false) const noexcept
	{
		return round_up ? check<true>(value) : check<false>(value);
	}

	inline constexpr operator T() const noexcept
	{
		return current;
	}
};

export
template<typename Rep, typename Period>
struct constraint<std::chrono::duration<Rep, Period>> final
{
	std::chrono::duration<Rep, Period> current, min, max, step;

	constexpr constraint() noexcept = default;

	constexpr constraint(
		std::chrono::duration<Rep, Period> current,
		std::chrono::duration<Rep, Period> min,
		std::chrono::duration<Rep, Period> max,
		std::chrono::duration<Rep, Period> step
	) noexcept
		: current(std::move(current)), min(std::move(min)), max(std::move(max)), step(std::move(step))
	{}

	template<std::integral T>
	constexpr constraint(const constraint<T>& other) :
		current(other.current), min(other.min), max(other.max), step(other.step)
	{}

	template<std::integral T>
	constexpr operator constraint<T>() const noexcept
	{
		return constraint<T>(current.count(), min.count(), max.count(), step.count());
	}

	template<bool round_up, typename Rep2, typename Period2>
	constexpr std::chrono::duration<Rep, Period> coerce(
		std::chrono::duration<Rep2, Period2> value
	) const noexcept
	{
		if (value < min)
			return min;
		if (value > max)
			return max;
		std::chrono::duration<Rep, Period> ret(value);
		if (auto r = (ret - min) % step)
			if constexpr (round_up)
				ret += step - r;
			else
				ret -= r;
		return ret;
	}

	template<typename Rep2, typename Period2>
	inline constexpr std::chrono::duration<Rep, Period> coerce(
		std::chrono::duration<Rep2, Period2> value,
		bool round_up = false
	) const noexcept
	{
		return round_up ? check<true>(std::move(value)) : check<false>(std::move(value));
	}

	inline constexpr operator const std::chrono::duration<Rep, Period>&() const & noexcept
	{
		return current;
	}

	inline constexpr operator std::chrono::duration<Rep, Period>() && noexcept
	{
		return std::move(current);
	}
};

export
template<std::floating_point T>
struct constraint<T> final
{
	T current, min, max;

	constexpr constraint() noexcept = default;

	constexpr constraint(T current, T min, T max) noexcept
		: current(current), min(min), max(max)
	{}

	inline constexpr T coerce(T value) const noexcept
	{
		return std::clamp<T>(value, min, max);
	}

	inline constexpr operator T() const noexcept
	{
		return current;
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
	_MVS_DEFINE_ENUM(device_type)
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
	_MVS_DEFINE_ENUM(exposure_auto_mode)
	{
		_MVS_EXPOSURE_AUTO_EXPAND(OFF),
		_MVS_EXPOSURE_AUTO_EXPAND(ONCE),
		_MVS_EXPOSURE_AUTO_EXPAND(CONTINUOUS)
	};
private:
#define _MVS_GAIN_MODE_EXPAND(N) _MVS_ENUM_EXPAND(GAIN_MODE_, , N, , , N)
public:
	_MVS_DEFINE_ENUM(gain_auto_mode)
	{
		_MVS_GAIN_MODE_EXPAND(OFF),
		_MVS_GAIN_MODE_EXPAND(ONCE),
		_MVS_GAIN_MODE_EXPAND(CONTINUOUS)
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
	device_type _type;

	device() noexcept : _handle(), _serial(), _type(device_type::UNKNOWN) {}

	template<typename CF>
	[[nodiscard]]
	inline bool _create(CF&& create, MV_CC_DEVICE_INFO *device_info) noexcept
	{
		if (_handle)
			return false;

		std::string s;
		auto t = static_cast<device_type>(device_info->nTLayerType);
		switch (t)
		{
			case device_type::GIG_E:
				s = reinterpret_cast<const char *>(device_info->SpecialInfo.stGigEInfo.chSerialNumber);
				break;
			case device_type::USB:
				s = reinterpret_cast<const char *>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber);
				break;
			case device_type::CAMERA_LINK:
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
	static std::vector<device> enumerate(device_type type) noexcept
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
	static std::vector<device> enumerate(device_type type, const std::string& log_path) noexcept
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
		other._type = device_type::UNKNOWN;
	}

	device& operator=(const device&) = delete;

	device& operator=(device&&) = delete;

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
	device_type type() const noexcept
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
			_type = device_type::UNKNOWN;
			return ret;
		}
		return true;
	}

	[[nodiscard]]
	bool open(access_mode mode = access_mode::EXCLUSIVE, unsigned short switch_over_key = 0) noexcept
	{
		return _wrap(MV_CC_OpenDevice, _handle, static_cast<unsigned int>(mode), switch_over_key);
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
#define _MVS_VALUE_SET_WRAP(PREFIX, SUFFIX, TYPE, DEFAULT_VALUE, FUNCTION, VALUE_STRING, CAST) \
	[[nodiscard]] \
	bool PREFIX##SUFFIX(TYPE value DEFAULT_VALUE) noexcept \
	{ \
		return _wrap(MV_CC_Set##FUNCTION, _handle, VALUE_STRING, CAST(value)); \
	}

#define _MVS_BOOL_WRAP(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_WRAP(enable, _##NAME, bool, = true, BoolValue, VALUE_STRING, ) \
	[[nodiscard]] \
	bool get_##NAME(bool& value) noexcept \
	{ \
		return _wrap(MV_CC_GetBoolValue, _handle, VALUE_STRING, &value); \
	}
public:
	_MVS_BOOL_WRAP(strobe, "StrobeEnable")

	_MVS_BOOL_WRAP(fps, "AcquisitionFrameRateEnable")
private:
#define _MVS_SIMPLE_REPLACE_SET_CALL(FUNCTION, NAME, TYPE, CAST) \
	[[nodiscard]] \
	bool set##NAME(TYPE value) noexcept \
	{ \
		return _wrap(MV_CC_Set##FUNCTION, _handle, CAST(value)); \
	}

#define _MVS_ENUM_WRAP(TYPE, VALUE_STRING) \
	_MVS_VALUE_SET_WRAP(set, , TYPE, , EnumValue, VALUE_STRING, static_cast<unsigned int>) \
	[[nodiscard]] \
	bool get(TYPE& value) noexcept \
	{ \
		if (MVCC_ENUMVALUE buffer; _wrap(MV_CC_GetEnumValue, _handle, VALUE_STRING, &buffer)) \
		{ \
			value = static_cast<TYPE>(buffer.nCurValue); \
			return true; \
		} \
		return false; \
	}
public:
	_MVS_ENUM_WRAP(exposure_auto_mode, "ExposureAuto")

	_MVS_ENUM_WRAP(gain_auto_mode, "GainAuto")

	_MVS_ENUM_WRAP(trigger_activation, "TriggerActivation")

	_MVS_ENUM_WRAP(trigger_mode, "TriggerMode")

	_MVS_ENUM_WRAP(trigger_source, "TriggerSource")

	_MVS_SIMPLE_REPLACE_SET_CALL(GrabStrategy, , grab_strategy, static_cast<MV_GRAB_STRATEGY>)
private:
#define _MVS_FLOAT_WRAP(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_WRAP(set, _##NAME, float, , FloatValue, VALUE_STRING, ) \
	[[nodiscard]] \
	bool get_##NAME(float& value) noexcept \
	{ \
		if (MVCC_FLOATVALUE buffer; _wrap(MV_CC_GetFloatValue, _handle, VALUE_STRING, &buffer)) \
		{ \
			value = buffer.fCurValue; \
			return true; \
		} \
		return false; \
	} \
	[[nodiscard]] \
	bool get_##NAME(constraint<float>& value) noexcept \
	{ \
		if (MVCC_FLOATVALUE buffer; _wrap(MV_CC_GetFloatValue, _handle, VALUE_STRING, &buffer)) \
		{ \
			value.current = buffer.fCurValue; \
			value.min = buffer.fMin; \
			value.max = buffer.fMax; \
			return true; \
		} \
		return false; \
	}
public:
	_MVS_FLOAT_WRAP(exposure, "ExposureTime")

	_MVS_FLOAT_WRAP(fps, "AcquisitionFrameRate")

	_MVS_FLOAT_WRAP(gain, "Gain")

	_MVS_FLOAT_WRAP(trigger_delay, "TriggerDelay")
private:
#define _MVS_INTEGER_WRAP(NAME, VALUE_STRING) \
	_MVS_VALUE_SET_WRAP(set, _##NAME, int64_t, , IntValue, VALUE_STRING, ) \
	[[nodiscard]] \
	bool get_##NAME(int64_t& value) noexcept \
	{ \
		if (MVCC_INTVALUE buffer; _wrap(MV_CC_GetIntValue, _handle, VALUE_STRING, &buffer)) \
		{ \
			value = buffer.nCurValue; \
			return true; \
		} \
		return false; \
	} \
	[[nodiscard]] \
	bool get_##NAME(constraint<int64_t>& value) noexcept \
	{ \
		if (MVCC_INTVALUE buffer; _wrap(MV_CC_GetIntValue, _handle, VALUE_STRING, &buffer)) \
		{ \
			value.current = buffer.nCurValue; \
			value.min = buffer.nMin; \
			value.max = buffer.nMax; \
			value.step = buffer.nInc; \
			return true; \
		} \
		return false; \
	}
public:
	_MVS_INTEGER_WRAP(line_debouncer, "LineDebouncerTime")

	_MVS_INTEGER_WRAP(strobe_line_delay, "StrobeLineDelay")

	_MVS_INTEGER_WRAP(strobe_line_duration, "StrobeLineDuration")

	[[nodiscard]]
	inline bool set_line_debouncer(const std::chrono::microseconds& value) noexcept
	{
		return set_line_debouncer(value.count());
	}

	[[nodiscard]]
	inline bool get_line_debouncer(std::chrono::microseconds& value) noexcept
	{
		if (int64_t buffer; get_line_debouncer(buffer))
		{
			value = std::chrono::microseconds(buffer);
			return true;
		}
		return false;
	}

	[[nodiscard]]
	inline bool get_line_debouncer(constraint<std::chrono::microseconds>& value) noexcept
	{
		if (constraint<int64_t> buffer; get_line_debouncer(buffer))
		{
			value = buffer;
			return true;
		}
		return false;
	}

	_MVS_SIMPLE_REPLACE_SET_CALL(ImageNodeNum, _cache_size, unsigned int, )

	_MVS_SIMPLE_REPLACE_SET_CALL(OutputQueueSize, _queue_size, unsigned int, )

	[[nodiscard]]
	bool set_receiver(queue& queue) noexcept
	{
		return _wrap(MV_CC_RegisterImageCallBackForBGR, _handle, queue::_callback, &queue);
	}
};

}
