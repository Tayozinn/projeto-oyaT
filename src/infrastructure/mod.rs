//! Agente 8 — Infraestrutura, Hardware e Estimativa de Recursos.
//!
//! Avalia capacidades locais (CPU, RAM, GPU/Vulkan) no Windows/Linux sem presumir
//! aceleração incompatível e fornece estimativas de memória para configurações neurais.

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub detected_gpu_hint: String,
    pub vulkan_available: bool,
    pub rocm_available: bool,
    pub recommended_training_device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub total_params: usize,
    pub weights_f32_mb: f64,
    pub adamw_states_mb: f64,
    pub activations_est_mb: f64,
    pub total_training_est_mb: f64,
    pub fits_16gb_ram: bool,
}

/// Detecta perfil básico de hardware local usando APIs padrão e sondagem segura de binários.
pub fn probe_hardware() -> HardwareProfile {
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let vulkan_available = command_exists("vulkaninfo");
    let rocm_available = command_exists("rocminfo") || std::env::var("ROCM_PATH").is_ok();

    let detected_gpu_hint = if cfg!(target_os = "windows") && vulkan_available {
        "GPU local compatível com Vulkan detectada (ex.: AMD Radeon RX 7600 8GB VRAM)".to_string()
    } else if rocm_available {
        "Ambiente ROCm detectado".to_string()
    } else {
        "Sem acelerador GPU dedicado detectado diretamente via CLI".to_string()
    };

    // Candle no Windows não oferece backend AMD/ROCm estável; execução em CPU multi-thread
    // é garantida e previsível para modelos experimentais (0.5M–30M parâmetros), enquanto
    // Vulkan/WGPU fica documentado como caminho futuro de aceleração GPU.
    let recommended_training_device = format!("CPU Multi-Thread ({} threads)", logical_cpus);

    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus,
        detected_gpu_hint,
        vulkan_available,
        rocm_available,
        recommended_training_device,
    }
}

/// Estima parâmetros e consumo de memória (pesos f32 + AdamW m/v + ativações) de um Transformer Decoder-only.
pub fn estimate_transformer_memory(
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_layers: usize,
    d_ff: usize,
    batch_size: usize,
) -> MemoryEstimate {
    let embed_params = vocab_size * d_model + max_seq_len * d_model;
    // Por camada: Q, K, V, O (4 * d_model^2) + FFN W1, W2 (2 * d_model * d_ff) + 2 RMSNorm (2 * d_model)
    let per_layer_params = 4 * d_model * d_model + 2 * d_model * d_ff + 2 * d_model;
    let lm_head_params = d_model * vocab_size + d_model;
    let total_params = embed_params + n_layers * per_layer_params + lm_head_params;

    let bytes_per_f32 = 4.0;
    let mb = 1024.0 * 1024.0;
    let weights_f32_mb = (total_params as f64 * bytes_per_f32) / mb;
    // AdamW guarda m + v + gradientes (~3x o tamanho dos pesos)
    let adamw_states_mb = weights_f32_mb * 3.0;
    // Estimativa conservadora de ativações por token em batch
    let act_elements = batch_size * max_seq_len * n_layers * (4 * d_model + d_ff);
    let activations_est_mb = (act_elements as f64 * bytes_per_f32) / mb;
    let total_training_est_mb = weights_f32_mb + adamw_states_mb + activations_est_mb;

    MemoryEstimate {
        total_params,
        weights_f32_mb,
        adamw_states_mb,
        activations_est_mb,
        total_training_est_mb,
        fits_16gb_ram: total_training_est_mb < 10_240.0,
    }
}

/// Micro-benchmark prático de multiplicação de matrizes na CPU local (retorna ms e GFLOP/s estimados).
pub fn benchmark_cpu_matmul(size: usize) -> (u128, f64) {
    let n = size.max(32);
    let a = vec![0.5f32; n * n];
    let b = vec![0.25f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let start = Instant::now();
    for i in 0..n {
        let a_row = &a[i * n..(i + 1) * n];
        let c_row = &mut c[i * n..(i + 1) * n];
        for (k, &av) in a_row.iter().enumerate() {
            let b_row = &b[k * n..(k + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let flops = 2.0 * (n as f64).powi(3);
    let gflops = (flops / secs) / 1e9;
    std::hint::black_box(&c);
    (elapsed.as_millis(), gflops)
}

fn command_exists(bin: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    let exts: &[&str] = if cfg!(target_os = "windows") {
        &["", ".exe", ".cmd", ".bat"]
    } else {
        &[""]
    };
    for dir in std::env::split_paths(&path_var) {
        for ext in exts {
            let candidate = dir.join(format!("{}{}", bin, ext));
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{benchmark_cpu_matmul, estimate_transformer_memory, probe_hardware};

    #[test]
    fn probes_hardware_without_panicking() {
        let hw = probe_hardware();
        assert!(hw.logical_cpus >= 1);
        assert!(!hw.os.is_empty());
    }

    #[test]
    fn estimates_memory_for_small_and_medium_models() {
        let small = estimate_transformer_memory(2048, 128, 128, 4, 512, 4);
        assert!(small.total_params > 500_000);
        assert!(small.fits_16gb_ram);

        let mid_15m = estimate_transformer_memory(4096, 256, 384, 6, 1536, 4);
        assert!(mid_15m.total_params > 10_000_000);
        assert!(mid_15m.fits_16gb_ram);
    }

    #[test]
    fn cpu_matmul_benchmark_runs() {
        let (_ms, gflops) = benchmark_cpu_matmul(64);
        assert!(gflops > 0.0);
    }
}
