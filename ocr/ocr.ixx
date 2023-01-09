module;

#include <cstddef>
#include <cstdint>

#include <concepts>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <clipper2/clipper.core.h>
#include <clipper2/clipper.offset.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

class model final
{
	Ort::Env _env;
	Ort::Session _session;
	std::vector<Ort::AllocatedStringPtr> _names;
	std::vector<const char *> _input_names, _output_names;
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

	model(const model&) = delete;

	model(model&&) noexcept = default;

	model& operator=(const model&) = delete;

	model& operator=(model&&) = delete;

	void operator()(
		const Ort::Value *inputs,
		size_t input_num,
		Ort::Value *outputs,
		size_t output_num,
		const Ort::RunOptions& run_options = {}
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
\
	inline void operator()(const Ort::Value& input, Ort::Value& output, const Ort::RunOptions& run_options = {})
	{
		operator()(&input, 1, &output, 1, run_options);
	}


	inline void operator()(
		const std::ranges::contiguous_range auto& inputs,
		std::ranges::contiguous_range auto& outputs,
		const Ort::RunOptions& run_options = {}
	)
	{
		operator()(
			std::ranges::data(inputs),
			std::ranges::size(inputs),
			std::ranges::data(outputs),
			std::ranges::size(outputs),
			run_options
		);
	}
};

}

namespace ocr
{

namespace
{

[[nodiscard]]
inline static Ort::SessionOptions _default_options() noexcept
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

}

namespace detectors
{

export
enum class algorithms
{
	DB
};

namespace
{

namespace preprocessors
{

class db final
{
	size_t _side_len;
	bool _min_side;
	cv::Scalar _mean, _stddev;
public:
	db(size_t side_len, bool min_side, cv::Scalar mean, cv::Scalar stddev) :
		_side_len(side_len),
		_min_side(min_side),
		_mean(std::move(mean)),
		_stddev(std::move(stddev))
	{}

	~db() noexcept = default;

	db(const db&) = default;

	db(db&&) noexcept = default;

	db& operator=(const db&) = delete;

	db& operator=(db&&) = delete;

	[[nodiscard]]
	inline cv::Mat operator()(const cv::Mat& src) const
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

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/data/imaug/operators.py#L86-L95

		cv::Mat ret;
		if (scaled)
		{
			cv::resize(src, ret, { width, height });
			ret.convertTo(ret, CV_32FC3, 1.0 / 255);
		}
		else
			src.convertTo(ret, CV_32FC3, 1.0 / 255);
		ret -= _mean;
		ret /= _stddev;
		return ret;
	}
};

}

namespace postprocessors
{

struct db final
{
	enum class scoring_method
	{
		BOX,
		APPROXIMATE,
		ORIGINAL
	};
private:
	double _threshold;
	bool _use_dilation;
	scoring_method _scoring_method;
	size_t _max_candidates;
	double _box_threshold;
	double _unclip_ratio;

	template<scoring_method M>
	[[nodiscard]]
	inline std::vector<cv::RotatedRect> _run(const cv::Mat& scores, const cv::Size& original_size) const
	{
		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L230-L235

		cv::Mat mask;
		cv::threshold(scores, mask, _threshold, 255, cv::ThresholdTypes::THRESH_BINARY);
		if (_use_dilation)
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
		for (size_t i = 0; i < contours.size() and kept < _max_candidates; ++i)
		{
			auto& contour = contours[i];

			if constexpr (M == scoring_method::APPROXIMATE)
			{
				cv::approxPolyDP(contour, contour, 0.002 * cv::arcLength(contour, true), true);
				if (contour.rows < 4)
					continue;
			}

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L182-L218

			auto bounding = cv::boundingRect(contour);
			cv::Mat mask(bounding.height, bounding.width, CV_8UC1, _BLACK);
			if constexpr (M == scoring_method::BOX)
			{
				auto enclosing = cv::minAreaRect(contour);
				cv::Mat vertices(4, 1, CV_32FC2);
				enclosing.points(vertices.ptr<cv::Point2f>());
				cv::fillPoly(mask, vertices, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());
			}
			else
				cv::fillPoly(mask, contour, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());

			if (cv::mean(scores(bounding), mask)[0] < _box_threshold)
				continue;

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L151-L157

			Clipper2Lib::Path64 contour_path;
			contour_path.reserve(contour.rows);
			for (int i = 0; i < contour.rows; ++i)
				contour_path.emplace_back(contour.at<Clipper2Lib::Point<int>>(i, 0));
			Clipper2Lib::ClipperOffset clipper_offset;
			clipper_offset.AddPaths({ contour_path }, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
			auto result = clipper_offset.Execute(
				cv::contourArea(contour) * _unclip_ratio / cv::arcLength(contour, true)
			);
			if (result.size() != 1)
				continue;

			const auto& unclipped = result.front();
			auto& new_contour = contours[kept++];
			new_contour.create(unclipped.size(), 1, CV_32SC2);
			for (size_t i = 0; i < unclipped.size(); ++i)
			{
				const auto& point = unclipped[i];
				new_contour.at<Clipper2Lib::Point<int>>(i, 0).Init(point.x, point.y);
			}
		}
		if (not kept)
			return {};
		contours.resize(kept);

		std::vector<cv::RotatedRect> ret;
		ret.reserve(kept);
		for (const auto& contour : contours)
			ret.push_back(cv::minAreaRect(contour));
		return ret;
	}
public:
	db(
		double threshold,
		bool use_dilation,
		scoring_method scoring_method,
		size_t max_candidates,
		double box_threshold,
		double unclip_ratio
	) :
		_threshold(threshold),
		_use_dilation(use_dilation),
		_scoring_method(scoring_method),
		_max_candidates(max_candidates),
		_box_threshold(box_threshold),
		_unclip_ratio(unclip_ratio)
	{}

	~db() noexcept = default;

	db(const db&) = default;

	db(db&&) noexcept = default;

	db& operator=(const db&) = delete;

	db& operator=(db&&) = delete;

	[[nodiscard]]
	inline std::vector<cv::RotatedRect> operator()(const cv::Mat& scores, const cv::Size& original_size) const
	{
#define _DB_PREPROCESSOR_TRANSFORM_FILTER(LABEL, METHOD) \
			LABEL: return _run<scoring_method::METHOD>(scores, original_size);

#define _DB_PREPROCESSOR_TRANSFORM_FILTER_DIRECT(METHOD) \
			_DB_PREPROCESSOR_TRANSFORM_FILTER(case scoring_method::METHOD, METHOD)

		switch (_scoring_method)
		{
			_DB_PREPROCESSOR_TRANSFORM_FILTER_DIRECT(BOX)
			_DB_PREPROCESSOR_TRANSFORM_FILTER_DIRECT(APPROXIMATE)
			_DB_PREPROCESSOR_TRANSFORM_FILTER(default, ORIGINAL)
		}
	}
};

}

template<algorithms>
struct processors;

template<>
struct processors<algorithms::DB> final
{
	using preprocessor = preprocessors::db;
	using postprocessor = postprocessors::db;
};

}

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual std::vector<cv::RotatedRect> operator()(const cv::Mat& image) const = 0;
};

export
template<algorithms algorithm>
class concrete final : public base
{
	processors<algorithm>::preprocessor _preprocessor;
	processors<algorithm>::postprocessor _postprocessor;
	model _model;
public:
	concrete(
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) : _model(model_path, options, graph_opt_level) {}

	concrete(
		const std::basic_string<ORTCHAR_T>& model_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) : _model(model_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level) {}

	virtual ~concrete() noexcept override = default;

	concrete(const concrete&) = delete;

	concrete(concrete&&) noexcept = default;

	concrete& operator=(const concrete&) = delete;

	concrete& operator=(concrete&&) = delete;

	virtual inline std::vector<cv::RotatedRect> operator()(const cv::Mat& image) const override
	{
		auto preprocessed = _preprocessor(image);
		auto input_tensor = _empty_tensor<float>(1, 3, preprocessed.rows, preprocessed.cols);
		auto data = input_tensor.GetTensorMutableData<float>();
		auto stride = preprocessed.rows * preprocessed.cols;
		cv::Mat split[] {
			{ preprocessed.rows, preprocessed.cols, CV_32FC1, data },
			{ preprocessed.rows, preprocessed.cols, CV_32FC1, data + stride },
			{ preprocessed.rows, preprocessed.cols, CV_32FC1, data + stride * 2 }
		};
		cv::split(preprocessed, split);

		auto output_tensor = _empty_tensor<float>(1, 1, preprocessed.rows, preprocessed.cols);
		_model(input_tensor, output_tensor);

		return _postprocessor(
			{ preprocessed.rows, preprocessed.cols, CV_32FC1, output_tensor.GetTensorMutableData<float>() },
			image.size()
		);
	}
};

export
using db = concrete<algorithms::DB>;

}

export
template<typename T>
concept detector = std::derived_from<T, detectors::base>;

export
template<detector Detector>
class system final
{
	std::optional<Detector> _detector;
};

namespace pmr
{

export
class system final
{
	std::unique_ptr<detectors::base> _detector;
};

}

}
