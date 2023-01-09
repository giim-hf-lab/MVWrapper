module;

#include <cstddef>
#include <cstdint>

#include <concepts>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <clipper2/clipper.core.h>
#include <clipper2/clipper.offset.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define _KEEP_MOVE_CONSTRUCTOR_ONLY(CLASS_NAME) \
	CLASS_NAME(const CLASS_NAME&) = delete; \
	CLASS_NAME(CLASS_NAME&&) noexcept = default; \
	CLASS_NAME& operator=(const CLASS_NAME&) = delete; \
	CLASS_NAME& operator=(CLASS_NAME&&) = delete;

export module ocr;

namespace std
{

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;

}

namespace
{

static Ort::AllocatorWithDefaultOptions _allocator {};

[[nodiscard]]
static inline Ort::Session _create_session(
	const std::basic_string<ORTCHAR_T>& model_path,
	Ort::Env& env,
	const Ort::SessionOptions& common_options,
	GraphOptimizationLevel graph_opt_level
)
{
	if (graph_opt_level > GraphOptimizationLevel::ORT_DISABLE_ALL)
	[[likely]]
	{
		auto optimised_model_path = model_path + ORT_TSTR(".opt");
		auto session_options = common_options.Clone();
		session_options
			.SetOptimizedModelFilePath(optimised_model_path.c_str())
			.SetGraphOptimizationLevel(graph_opt_level);
		return { env, model_path.c_str(), session_options };
	}
	return { env, model_path.c_str(), common_options };
}

template<std::arithmetic T>
static inline Ort::Value _empty_tensor(std::integral auto... dimensions)
{
	int64_t shapes[] { int64_t(dimensions)... };
	return Ort::Value::CreateTensor<T>(_allocator, shapes, sizeof...(dimensions));
}

static const Ort::RunOptions _DEFAULT_RUN_OPTIONS {};

class model final
{
	Ort::Env _env;
	Ort::Session _session;
	std::vector<Ort::AllocatedStringPtr> _names;
	std::vector<const char *> _input_names, _output_names;

	inline void _run(
		const Ort::Value *inputs,
		size_t input_num,
		Ort::Value *outputs,
		size_t output_num,
		const Ort::RunOptions& run_options = _DEFAULT_RUN_OPTIONS
	)
	{
		if (input_num != _input_names.size())
		[[unlikely]]
			throw std::runtime_error("model input number mismatch");
		if (output_num != _output_names.size())
		[[unlikely]]
			throw std::runtime_error("model output number mismatch");
		_session.Run(run_options, _input_names.data(), inputs, input_num, _output_names.data(), outputs, output_num);
	}
public:
	model(
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& common_options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_env(),
		_session(_create_session(model_path, _env, common_options, graph_opt_level)),
		_names(),
		_input_names(),
		_output_names()
	{
		auto input_num = _session.GetInputCount(), output_num = _session.GetOutputCount();

		_names.reserve(input_num + output_num);
		_input_names.reserve(input_num);
		for (size_t i = 0; i < input_num; ++i)
			_input_names.emplace_back(_names.emplace_back(_session.GetInputNameAllocated(i, _allocator)).get());
		_output_names.reserve(output_num);
		for (size_t i = 0; i < output_num; ++i)
			_output_names.emplace_back(_names.emplace_back(_session.GetOutputNameAllocated(i, _allocator)).get());
	}

	~model() noexcept = default;

	_KEEP_MOVE_CONSTRUCTOR_ONLY(model)

	inline void run(
		const Ort::Value& input,
		Ort::Value& output,
		const Ort::RunOptions& run_options = _DEFAULT_RUN_OPTIONS
	)
	{
		_run(&input, 1, &output, 1, run_options);
	}

	inline void run(
		const std::ranges::contiguous_range auto& inputs,
		std::ranges::contiguous_range auto& outputs,
		const Ort::RunOptions& run_options = _DEFAULT_RUN_OPTIONS
	)
	{
		_run(
			std::ranges::data(inputs),
			std::ranges::size(inputs),
			std::ranges::data(outputs),
			std::ranges::size(outputs),
			run_options
		);
	}
};

}

namespace cv
{

template<typename>
struct depths final
{
	static constexpr inline int make(int channels) noexcept;
};

#define _CV_SUPP_MAKE_DEPTHS(TYPE, VALUE) \
constexpr inline int depths<TYPE>::make(int channels) noexcept \
{ \
	return CV_MAKETYPE(VALUE, channels); \
}

_CV_SUPP_MAKE_DEPTHS(uint8_t, CV_8U)

_CV_SUPP_MAKE_DEPTHS(int8_t, CV_8S)

_CV_SUPP_MAKE_DEPTHS(uint16_t, CV_16U)

_CV_SUPP_MAKE_DEPTHS(int16_t, CV_16S)

_CV_SUPP_MAKE_DEPTHS(int32_t, CV_32S)

_CV_SUPP_MAKE_DEPTHS(float, CV_32F)

_CV_SUPP_MAKE_DEPTHS(double, CV_64F)

}

namespace ocr
{

namespace
{

template<std::arithmetic PointType, size_t... Indices, std::arithmetic... PointTypes>
[[nodiscard]]
static inline cv::Mat _points_to_mat_impl(std::index_sequence<Indices...>, const cv::Point_<PointTypes>&... points)
{
	cv::Mat output(sizeof...(PointTypes), 1, cv::depths<PointType>::make(2));
	((output.at<cv::Point_<PointType>>(Indices, 0) = points), ...);
	return output;
}

template<std::arithmetic PointType, std::arithmetic... PointTypes>
[[nodiscard]]
static inline cv::Mat _points_to_mat(const cv::Point_<PointTypes>&... points)
{
	return _points_to_mat_impl<PointType>(std::index_sequence_for<PointTypes...> {}, points...);
}

[[nodiscard]]
static inline Ort::SessionOptions _default_options() noexcept
{
	Ort::SessionOptions session_options;
	session_options
		.EnableCpuMemArena()
		.EnableMemPattern()
		.DisableProfiling()
		.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions {});
	return session_options;
}

static const auto _GLOBAL_DEFAULT_OPTIONS = _default_options();

static const auto _BLACK = cv::Scalar::all(0), _WHITE = cv::Scalar::all(255);

static const auto _kernel_2x2 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, { 2, 2 });

static inline void _split_channels(const cv::Mat& mat, float *pos)
{
	auto stride = mat.rows * mat.cols;
	std::vector<cv::Mat> channels;
	channels.reserve(mat.channels());
	for (auto i = 0; i < mat.channels(); ++i)
		channels.emplace_back(mat.rows, mat.cols, CV_32FC1, pos + i * stride);
	cv::split(mat, channels.data());
}

}

namespace scalers
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, cv::Mat& dest) = 0;
};

export
class resize final : public base
{
	cv::Size _size;
public:
	resize(cv::Size size) noexcept : _size(std::move(size)) {}

	virtual ~resize() noexcept override = default;

	_KEEP_MOVE_CONSTRUCTOR_ONLY(resize)

	[[nodiscard]]
	virtual inline bool run(const cv::Mat& src, cv::Mat& dest) override
	{
		if (src.size() == _size)
			return false;
		cv::resize(src, dest, _size);
		return true;
	}
};

export
class zoom final : public base
{
	size_t _side_len;
	bool _min_side;
public:
	zoom(size_t side_len, bool min_side = true) noexcept : _side_len(side_len), _min_side(min_side) {}

	virtual ~zoom() noexcept override = default;

	_KEEP_MOVE_CONSTRUCTOR_ONLY(zoom)

	[[nodiscard]]
	virtual inline bool run(const cv::Mat& src, cv::Mat& dest) override
	{
		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/data/imaug/operators.py#L254-L301

		auto height = src.rows, width = src.cols;
		bool scaled;
		if (
			_min_side and height <= width and height < _side_len or
			not _min_side and height >= width and height > _side_len
			)
		{
			width *= double(_side_len) / height;
			height = _side_len;
			scaled = true;
		}
		else if (
			_min_side and height > width and width < _side_len or
			not _min_side and height < width and width > _side_len
			)
		{
			height *= double(_side_len) / width;
			width = _side_len;
			scaled = true;
		}
		else
			scaled = false;
		if (auto r = height % 32)
		{
			height += 32 - r;
			scaled = true;
		}
		if (auto r = width % 32)
		{
			width += 32 - r;
			scaled = true;
		}

		if (scaled)
			cv::resize(src, dest, { width, height });
		return scaled;
	}
};

}

template<typename T>
concept scaler = std::derived_from<T, scalers::base>;

namespace detectors
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(const cv::Mat& image, std::vector<cv::RotatedRect>& boxes) = 0;
};

export
struct trivial final : base
{
	[[nodiscard]]
	virtual inline bool run(const cv::Mat& image, std::vector<cv::RotatedRect>& boxes) override
	{
		return false;
	}
};

export
enum class algorithms
{
	DB
};

export
template<algorithms>
struct concrete final : public base
{
	struct parameters;
private:
	parameters _parameters;
	model _model;

	inline void _extract(const cv::Mat& scores, std::vector<cv::RotatedRect>& boxes) const;
public:
	concrete(
		parameters parameters,
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_parameters(std::move(parameters)),
		_model(model_path, options, graph_opt_level)
	{}

	concrete(
		parameters parameters,
		const std::basic_string<ORTCHAR_T>& model_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_parameters(std::move(parameters)),
		_model(model_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level)
	{}

	virtual ~concrete() noexcept override = default;

	_KEEP_MOVE_CONSTRUCTOR_ONLY(concrete)

	[[nodiscard]]
	virtual inline bool run(const cv::Mat& image, std::vector<cv::RotatedRect>& boxes) override
	{
		auto input_tensor = _empty_tensor<float>(1, 3, image.rows, image.cols);
		_split_channels(image, input_tensor.GetTensorMutableData<float>());

		auto output_tensor = _empty_tensor<float>(1, 1, image.rows, image.cols);
		_model.run(input_tensor, output_tensor);

		_extract({ image.rows, image.cols, CV_32FC1, output_tensor.GetTensorMutableData<float>() }, boxes);
		return true;
	}
};

export
using db = concrete<algorithms::DB>;

struct db::parameters final
{
	enum class scoring_methods
	{
		BOX,
		APPROXIMATE,
		ORIGINAL
	};

	double threshold;
	bool use_dilation;
	scoring_methods scoring_method;
	size_t max_candidates;
	double box_threshold;
	double unclip_ratio;

	parameters(
		double threshold,
		bool use_dilation,
		scoring_methods scoring_method,
		size_t max_candidates,
		double box_threshold,
		double unclip_ratio
	) noexcept :
		threshold(threshold),
		use_dilation(use_dilation),
		scoring_method(scoring_method),
		max_candidates(max_candidates),
		box_threshold(box_threshold),
		unclip_ratio(unclip_ratio)
	{}

	~parameters() noexcept = default;

	parameters(const parameters&) noexcept = default;

	parameters(parameters&&) noexcept = default;

	parameters& operator=(const parameters&) noexcept = default;

	parameters& operator=(parameters&&) noexcept = default;
};

namespace
{

template<db::parameters::scoring_methods scoring_method>
[[nodiscard]]
static inline bool _db_transform(
	const cv::Mat& scores,
	cv::Mat& src,
	cv::Mat& dest,
	double box_threshold,
	double unclip_ratio
)
{
	if constexpr (scoring_method == db::parameters::scoring_methods::APPROXIMATE)
	{
		cv::approxPolyDP(src, src, 0.002 * cv::arcLength(src, true), true);
		if (dest.rows < 4)
			return false;
	}

	// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L182-L218

	auto bounding = cv::boundingRect(src);
	cv::Mat mask(bounding.height, bounding.width, CV_8UC1, _BLACK);
	if constexpr (scoring_method == db::parameters::scoring_methods::BOX)
	{
		auto enclosing = cv::minAreaRect(src);
		cv::Mat vertices(4, 1, CV_32FC2);
		enclosing.points(vertices.ptr<cv::Point2f>());
		cv::fillPoly(mask, vertices, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());
	}
	else
		cv::fillPoly(mask, src, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());

	if (cv::mean(scores(bounding), mask)[0] < box_threshold)
		return false;

	// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L151-L157

	Clipper2Lib::Path64 contour_path;
	contour_path.reserve(src.rows);
	for (int i = 0; i < src.rows; ++i)
		contour_path.emplace_back(src.at<Clipper2Lib::Point<int>>(i, 0));
	Clipper2Lib::ClipperOffset clipper_offset;
	clipper_offset.AddPaths({ contour_path }, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
	auto result = clipper_offset.Execute(cv::contourArea(src) * unclip_ratio / cv::arcLength(src, true));
	if (result.size() != 1)
		return false;

	const auto& unclipped = result.front();
	dest.create(unclipped.size(), 1, CV_32SC2);
	for (size_t i = 0; i < unclipped.size(); ++i)
	{
		const auto& point = unclipped[i];
		dest.at<Clipper2Lib::Point<int>>(i, 0).Init(point.x, point.y);
	}
	return true;
}

}

inline void db::_extract(const cv::Mat& scores, std::vector<cv::RotatedRect>& boxes) const
{
	// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L230-L235

	cv::Mat mask;
	cv::threshold(scores, mask, _parameters.threshold, 255, cv::ThresholdTypes::THRESH_BINARY);
	if (_parameters.use_dilation)
		cv::dilate(mask, mask, _kernel_2x2);

	// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L57-L149

	std::vector<cv::Mat> contours;
	cv::findContours(
		mask,
		contours,
		cv::RetrievalModes::RETR_LIST,
		cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE
	);

	size_t kept = 0;

#define _OCR_DB_POSTPROCESS_TRANSFORM(LABEL, METHOD) \
		LABEL: \
			for (size_t i = 0; i < contours.size() and kept < _parameters.max_candidates; ++i) \
				if (_db_transform<parameters::scoring_methods::METHOD>( \
					scores, \
					contours[i], \
					contours[kept], \
					_parameters.box_threshold, \
					_parameters.unclip_ratio \
				)) \
					++kept;

#define _OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(METHOD) \
		_OCR_DB_POSTPROCESS_TRANSFORM(case parameters::scoring_methods::METHOD, METHOD) \
			break;

	switch (_parameters.scoring_method)
	{
		_OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(BOX)
		_OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(APPROXIMATE)
		_OCR_DB_POSTPROCESS_TRANSFORM(default, ORIGINAL)
	}

	if (not kept)
		return;
	contours.resize(kept);

	boxes.reserve(boxes.size() + kept);
	for (const auto& contour : contours)
		boxes.push_back(cv::minAreaRect(contour));
}

}

export
template<typename T>
concept detector = std::derived_from<T, detectors::base>;

namespace classifiers
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual inline std::vector<size_t> run(const std::vector<cv::Mat>& fragments) = 0;
};

export
struct trivial final : public base
{
	[[nodiscard]]
	virtual inline std::vector<size_t> run(const std::vector<cv::Mat>& fragments) override
	{
		return {};
	}
};

export
struct concrete final : public base
{
	static const cv::Scalar _mean, _stddev;

	struct parameters final
	{
		size_t batch_size;
		cv::Size shape;
		double threshold;

		parameters(
			size_t batch_size,
			cv::Size shape,
			double threshold
		) :
			batch_size(batch_size),
			shape(std::move(shape)),
			threshold(threshold)
		{}

		~parameters() noexcept = default;

		parameters(const parameters&) noexcept = default;

		parameters(parameters&&) noexcept = default;

		parameters& operator=(const parameters&) noexcept = default;

		parameters& operator=(parameters&&) noexcept = default;
	};
private:
	parameters _parameters;
	model _model;
public:
	concrete(
		parameters parameters,
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_parameters(std::move(parameters)),
		_model(model_path, options, graph_opt_level)
	{}

	concrete(
		parameters parameters,
		const std::basic_string<ORTCHAR_T>& model_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_parameters(std::move(parameters)),
		_model(model_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level)
	{}

	virtual ~concrete() noexcept override = default;

	_KEEP_MOVE_CONSTRUCTOR_ONLY(concrete)

	[[nodiscard]]
	virtual inline std::vector<size_t> run(const std::vector<cv::Mat>& fragments) override
	{
		auto input_tensor = _empty_tensor<float>(
			_parameters.batch_size,
			3,
			_parameters.shape.height,
			_parameters.shape.width
		);
		auto output_tensor = _empty_tensor<float>(_parameters.batch_size, 2);
		auto stride = _parameters.shape.area() * 3;
		auto write_pos = input_tensor.GetTensorMutableData<float>();
		std::vector<size_t> ret;
		ret.reserve(fragments.size());
		for (size_t i = 0; i < fragments.size(); i += _parameters.batch_size)
		{
			size_t current_batch = std::min(_parameters.batch_size, fragments.size() - i);
			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				cv::Mat tmp;
				if (const auto& fragment = fragments[j]; fragment.size() == _parameters.shape)
					fragment.convertTo(tmp, CV_32FC3);
				else
				{
					cv::resize(fragment, tmp, _parameters.shape);
					tmp.convertTo(tmp, CV_32FC3);
				}
				tmp -= _mean;
				tmp /= _stddev;
				_split_channels(tmp, write_pos + c * stride);
			}
			std::fill_n(write_pos + current_batch * stride, (_parameters.batch_size - current_batch) * stride, 0.0f);

			_model.run(input_tensor, output_tensor);
			cv::Mat scores(current_batch, 2, CV_32FC1, output_tensor.GetTensorMutableData<float>());
			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				double score;
				int index;
				cv::minMaxIdx(scores.row(c), nullptr, &score, nullptr, &index);
				if (index and score >= _parameters.threshold)
					ret.push_back(j);
			}
		}
		return ret;
	}
};

const auto concrete::_mean = cv::Scalar::all(0.5), concrete::_stddev = cv::Scalar::all(0.5);

}

export
template<typename T>
concept classifier = std::derived_from<T, classifiers::base>;

namespace recognisers
{

struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual std::vector<std::tuple<size_t, std::string, double>> run(const std::vector<cv::Mat>& fragments) = 0;
};

export
enum class algorithms
{
	CTC
};

}

export
template<typename T>
concept recogniser = std::derived_from<T, recognisers::base>;

export
template<
	scaler Scaler,
	detector Detector,
	classifier Classifier,
	recogniser Recogniser
>
class system final
{
	Scaler _scaler;
	Detector _detector;
	Classifier _classifier;
	Recogniser _recogniser;
};

namespace pmr
{

export
class system final
{
	std::unique_ptr<scalers::base> _scaler;
	std::unique_ptr<detectors::base> _detector;
	std::unique_ptr<classifiers::base> _classifier;
	std::unique_ptr<recognisers::base> _recogniser;
};

}

}
