#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#endif

namespace deep_gemm::warmup {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

// Warmup kernels by generating code for each M value, deduplicating by code string,
// and compiling only unique kernels. Returns the number of unique kernels compiled.
static int warmup_kernels(
    const std::string& kernel_type,
    const std::vector<int>& m_list,
    const int& n,
    const int& k,
    const int& num_groups,
    const std::string& compiled_dims = "nk")
{
    const auto arch_major = device_runtime->get_arch_major();
    const auto num_sms = device_runtime->get_num_sms();

    auto t_start = std::chrono::steady_clock::now();

    fprintf(stderr, "[warmup] kernel_type=%s N=%d K=%d G=%d, processing %zu M values (range %d-%d)\n",
            kernel_type.c_str(), n, k, num_groups,
            m_list.size(),
            m_list.empty() ? 0 : m_list.front(),
            m_list.empty() ? 0 : m_list.back());

    // Map from code string -> kernel name for deduplication
    std::unordered_map<std::string, std::string> unique_kernels;

    // Zero-initialized CUtensorMap (never read by generate_impl)
    CUtensorMap zero_tm{};
    memset(&zero_tm, 0, sizeof(zero_tm));

    // Keep references alive for Args structs
    const std::optional<std::string> no_epilogue = std::nullopt;

    for (const int& m : m_list) {
        if (m <= 0) continue;

        std::string code;
        std::string kernel_name;

        if (kernel_type == "fp8_gemm_nt") {
            // Normal FP8 GEMM: a=fp8, b=fp8, cd=float, with_accumulation=true
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::Normal, KernelType::Kernel1D1D,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kFloat, true, num_sms);
                const SM90FP8Gemm1D1DRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = 1,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .gmem_a_ptr = nullptr, .gmem_b_ptr = nullptr,
                    .grouped_layout = nullptr, .tensor_map_buffer = nullptr,
                    .tensor_map_a_base = zero_tm, .tensor_map_b_base = zero_tm,
                    .tensor_map_sfa = zero_tm, .tensor_map_sfb = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM90FP8Gemm1D1DRuntime::generate(args);
                kernel_name = "sm90_fp8_gemm_1d1d";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::Normal, KernelType::Kernel1D1D,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kFloat, true, num_sms);
                const SM100FP8FP4Gemm1D1DRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = 1,
                    .gran_k_a = 128, .gran_k_b = 128,
                    .compiled_dims = compiled_dims,
                    .epilogue_type = no_epilogue,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_sfa = zero_tm, .tensor_map_sfb = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
                kernel_name = "sm100_fp8_fp4_gemm_1d1d";
            }
        } else if (kernel_type == "m_grouped_fp8_gemm_nt_masked") {
            // Masked grouped FP8 GEMM: a=fp8, b=fp8, cd=bf16, with_accumulation=false
            // m_list values are expected_m values
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::MGroupedMasked, KernelType::Kernel1D2D,
                    m, n, k, num_groups, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kBFloat16, false, num_sms);
                const SM90FP8Gemm1D2DRuntime::Args args = {
                    .major_sfb = cute::UMMA::Major::K,
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .epilogue_type = no_epilogue,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .sfb = nullptr, .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_d = zero_tm, .tensor_map_sfa = zero_tm,
                };
                code = SM90FP8Gemm1D2DRuntime::generate(args);
                kernel_name = "sm90_fp8_m_grouped_gemm_masked_1d2d";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::MGroupedMasked, KernelType::Kernel1D1D,
                    m, n, k, num_groups, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kBFloat16, false, num_sms);
                const SM100FP8FP4Gemm1D1DRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .gran_k_a = 128, .gran_k_b = 128,
                    .compiled_dims = compiled_dims,
                    .epilogue_type = no_epilogue,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_sfa = zero_tm, .tensor_map_sfb = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
                kernel_name = "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d";
            }
        } else if (kernel_type == "m_grouped_fp8_gemm_nt_contiguous") {
            // Contiguous grouped FP8 GEMM: a=fp8, b=fp8, cd=bf16, with_accumulation=false
            // get_best_config uses num_groups=1 (contiguous layout is treated as a whole)
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::MGroupedContiguous, KernelType::Kernel1D2D,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kBFloat16, false, num_sms);
                const SM90FP8Gemm1D2DRuntime::Args args = {
                    .major_sfb = cute::UMMA::Major::K,
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .epilogue_type = no_epilogue,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .sfb = nullptr, .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_d = zero_tm, .tensor_map_sfa = zero_tm,
                };
                code = SM90FP8Gemm1D2DRuntime::generate(args);
                kernel_name = "sm90_m_grouped_fp8_gemm_contiguous_1d2d";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn,
                    torch::kBFloat16, false, num_sms);
                const SM100FP8FP4Gemm1D1DRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .gran_k_a = 128, .gran_k_b = 128,
                    .compiled_dims = compiled_dims,
                    .epilogue_type = no_epilogue,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_sfa = zero_tm, .tensor_map_sfb = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
                kernel_name = "sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d";
            }
        } else if (kernel_type == "bf16_gemm_nt") {
            // Normal BF16 GEMM: a=bf16, b=bf16, cd=float, with_accumulation=false
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::Normal, KernelType::KernelNoSF,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kFloat, false, num_sms);
                const SM90BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = 1,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM90BF16GemmRuntime::generate(args);
                kernel_name = "sm90_bf16_gemm";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::Normal, KernelType::KernelNoSF,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kFloat, false, num_sms);
                const SM100BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = 1,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100BF16GemmRuntime::generate(args);
                kernel_name = "sm100_bf16_gemm";
            }
        } else if (kernel_type == "m_grouped_bf16_gemm_nt_masked") {
            // Masked grouped BF16 GEMM: a=bf16, b=bf16, cd=bf16, with_accumulation=false
            // m_list values are expected_m values
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::MGroupedMasked, KernelType::KernelNoSF,
                    m, n, k, num_groups, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kBFloat16, false, num_sms);
                const SM90BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM90BF16GemmRuntime::generate(args);
                kernel_name = "sm90_bf16_m_grouped_gemm_masked";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::MGroupedMasked, KernelType::KernelNoSF,
                    m, n, k, num_groups, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kBFloat16, false, num_sms);
                const SM100BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100BF16GemmRuntime::generate(args);
                kernel_name = "sm100_bf16_m_grouped_gemm_masked";
            }
        } else if (kernel_type == "m_grouped_bf16_gemm_nt_contiguous") {
            // Contiguous grouped BF16 GEMM: a=bf16, b=bf16, cd=bf16, with_accumulation=false
            // get_best_config uses num_groups=1
            if (arch_major == 9) {
                const auto& config = get_best_config<SM90ArchSpec>(
                    GemmType::MGroupedContiguous, KernelType::KernelNoSF,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kBFloat16, false, num_sms);
                const SM90BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM90BF16GemmRuntime::generate(args);
                kernel_name = "sm90_m_grouped_bf16_gemm_contiguous";
            } else if (arch_major == 10) {
                const auto& config = get_best_config<SM100ArchSpec>(
                    GemmType::MGroupedContiguous, KernelType::KernelNoSF,
                    m, n, k, 1, cute::UMMA::Major::K, cute::UMMA::Major::K,
                    torch::kBFloat16, torch::kBFloat16,
                    torch::kBFloat16, false, num_sms);
                const SM100BF16GemmRuntime::Args args = {
                    .m = m, .n = n, .k = k, .num_groups = num_groups,
                    .compiled_dims = compiled_dims,
                    .gemm_config = config,
                    .launch_args = LaunchArgs(1, 1),
                    .grouped_layout = nullptr,
                    .tensor_map_a = zero_tm, .tensor_map_b = zero_tm,
                    .tensor_map_cd = zero_tm,
                };
                code = SM100BF16GemmRuntime::generate(args);
                kernel_name = "sm100_bf16_m_grouped_gemm_contiguous";
            }
        } else {
            DG_HOST_UNREACHABLE("Unknown kernel type: " + kernel_type);
        }

        if (not code.empty()) {
            unique_kernels.emplace(std::move(code), std::move(kernel_name));
        }
    }

    auto t_dedup = std::chrono::steady_clock::now();
    double dedup_secs = std::chrono::duration<double>(t_dedup - t_start).count();

    const int num_unique = static_cast<int>(unique_kernels.size());
    fprintf(stderr, "[warmup] Deduplication done in %.1fs: %zu M values -> %d unique kernels\n",
            dedup_secs, m_list.size(), num_unique);

    // Compile each unique kernel (disk only, no GPU load)
    int compiled = 0;
    int cached = 0;
    for (const auto& [code, name] : unique_kernels) {
        compiled++;
        auto t_kernel = std::chrono::steady_clock::now();
        bool was_cached = compiler->ensure_compiled(name, code);
        double kernel_secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_kernel).count();

        if (was_cached) {
            cached++;
            fprintf(stderr, "[warmup] [%d/%d] %s already cached (%.3fs)\n",
                    compiled, num_unique, name.c_str(), kernel_secs);
        } else {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_dedup).count();
            int remaining = num_unique - compiled;
            int actually_compiled = compiled - cached;
            double avg = actually_compiled > 0 ? elapsed / actually_compiled : 0;
            double eta = avg * remaining;
            fprintf(stderr, "[warmup] [%d/%d] Compiled %s in %.1fs (avg %.1fs/kernel, ETA %.0fs)\n",
                    compiled, num_unique, name.c_str(), kernel_secs, avg, eta);
        }
    }

    double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    fprintf(stderr, "[warmup] Completed %s: %d unique kernels (%d compiled, %d cached) in %.1fs\n",
            kernel_type.c_str(), num_unique, num_unique - cached, cached, total_secs);

    return num_unique;
}

#endif // DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("warmup_kernels", &warmup_kernels,
          py::arg("kernel_type"),
          py::arg("m_list"),
          py::arg("n"),
          py::arg("k"),
          py::arg("num_groups"),
          py::arg("compiled_dims") = "nk",
          py::call_guard<py::gil_scoped_release>());
#endif
}

} // namespace deep_gemm::warmup
