use crate::cli::error::CliResult;
use std::io::Write;

pub(super) fn handle_hw_command(out: &mut dyn Write) {
    let _ = run_hw_cli(out);
}

pub fn run_hw_cli(out: &mut dyn Write) -> CliResult<()> {
    let hw = crate::platform::hardware::probe_hardware();
    let (ms, gflops) = crate::platform::hardware::benchmark_cpu_matmul(192);
    let mem_10m = crate::platform::hardware::estimate_transformer_memory_with_heads(
        4096, 256, 384, 6, 6, 1536, 1,
    );
    let _ = writeln!(out, "=== Hardware & Infraestrutura ServerMind ===");
    let _ = writeln!(out, "OS / Arch        : {} / {}", hw.os, hw.arch);
    let _ = writeln!(out, "CPUs lógicas     : {}", hw.logical_cpus);
    if let (Some(tot_mb), Some(avail_mb)) = (hw.total_system_ram_mb, hw.available_system_ram_mb) {
        let prof = hw
            .recommended_runtime_profile
            .as_deref()
            .unwrap_or("standard");
        let _ = writeln!(
            out,
            "RAM Física Host  : {} MB total ({} MB livres) | perfil={}",
            tot_mb, avail_mb, prof
        );
    }
    let _ = writeln!(out, "GPU hint         : {}", hw.detected_gpu_hint);
    let _ = writeln!(
        out,
        "Vulkan / ROCm    : {} / {}",
        hw.vulkan_available, hw.rocm_available
    );
    let _ = writeln!(out, "Dispositivo ativo: {}", hw.recommended_training_device);
    let _ = writeln!(
        out,
        "Benchmark CPU    : 192x192 em {} ms ({:.2} GFLOPS)",
        ms, gflops
    );
    if let Some((gpu_ms, gpu_gflops)) = crate::platform::hardware::benchmark_gpu_matmul(512) {
        let _ = writeln!(
            out,
            "Benchmark GPU    : 512x512 VRAM em {:.2} ms ({:.2} GFLOPS)",
            gpu_ms, gpu_gflops
        );
    }
    let _ = writeln!(
        out,
        "Estimativa ~14M  : {:.2} M params | pesos_f32={:.1} MB (fp16={:.1} | int8={:.1} | int4={:.1} MB) | treino_seq={:.1} MB | infer_kv(f32={:.1} | fp16={:.1} | int8={:.1} MB) | cabe_8gb_ram={} | cabe_rx7600_8gb={}",
        mem_10m.total_params as f64 / 1e6,
        mem_10m.weights_f32_mb,
        mem_10m.weights_fp16_mb,
        mem_10m.weights_int8_mb,
        mem_10m.weights_int4_mb,
        mem_10m.total_training_est_mb,
        mem_10m.inference_kv_fp32_mb,
        mem_10m.inference_kv_fp16_mb,
        mem_10m.inference_kv_int8_mb,
        mem_10m.fits_8gb_ram_inference,
        mem_10m.fits_8gb_vram_rx7600
    );
    Ok(())
}
