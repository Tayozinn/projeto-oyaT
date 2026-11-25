//! Agente 8 — Infraestrutura, Hardware e Estimativa de Recursos.
//!
//! Avalia capacidades locais (CPU, RAM, GPU/Vulkan) no Windows/Linux sem presumir
//! aceleração incompatível e fornece estimativas de memória para configurações neurais.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Limite superior (em MiB) para classificar automaticamente um computador como perfil `LowEnd8Gb`
/// (8 GiB = 8192 MiB; margem até 8704 MiB cobre variações de reserva de BIOS/vídeo integrado).
pub const LOW_END_SYSTEM_RAM_THRESHOLD_MB: u64 = 8_704;

/// Teto seguro padrão de RAM dedicada ao motor neural em máquinas de 8 GB RAM (`2.048 MiB`),
/// preservando `~6 GB` livres para o sistema operacional, serviços e terminal.
pub const DEFAULT_LOW_END_INFERENCE_BUDGET_MB: usize = 2_048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeProfileKind {
    Standard,
    LowEnd8Gb,
}

impl RuntimeProfileKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::LowEnd8Gb => "low-end-8gb",
        }
    }
}

/// Orçamento preventivo de recursos de runtime (`SM-008` — Etapa 9) para evitar `OOM`
/// e paginação excessiva (*swap thrashing*) em computadores modestos de 8 GB de RAM.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RuntimeBudget {
    pub profile: RuntimeProfileKind,
    /// Limite máximo opcional em bytes para `pesos + KvCache + buffers de prefill` na inferência.
    pub max_inference_ram_bytes: Option<usize>,
    /// Precisão recomendada para o `KvCache` neste perfil.
    pub default_kv_precision: crate::neural::KvCachePrecision,
    /// Limite opcional de threads ativas no `CpuThreadPool`.
    pub max_threads: Option<usize>,
}

impl Default for RuntimeBudget {
    fn default() -> Self {
        Self::auto_detect()
    }
}

impl RuntimeBudget {
    /// Detecta automaticamente o orçamento apropriado com base na RAM física do host.
    pub fn auto_detect() -> Self {
        if let Some((total_mb, _avail_mb)) = probe_system_ram_mb() {
            if total_mb <= LOW_END_SYSTEM_RAM_THRESHOLD_MB {
                return Self::low_end_8gb();
            }
        }
        Self {
            profile: RuntimeProfileKind::Standard,
            max_inference_ram_bytes: None,
            default_kv_precision: crate::neural::KvCachePrecision::Fp32,
            max_threads: None,
        }
    }

    /// Perfil estrito para distribuição e execução em PCs modestos (`8 GB RAM`, sem GPU dedicada).
    pub fn low_end_8gb() -> Self {
        Self {
            profile: RuntimeProfileKind::LowEnd8Gb,
            max_inference_ram_bytes: Some(DEFAULT_LOW_END_INFERENCE_BUDGET_MB * 1024 * 1024),
            default_kv_precision: crate::neural::KvCachePrecision::Fp16,
            max_threads: None,
        }
    }

    pub fn with_max_ram_mb(mut self, max_mb: usize) -> Self {
        self.max_inference_ram_bytes = Some(max_mb.saturating_mul(1024 * 1024));
        self
    }

    pub fn with_max_threads(mut self, threads: usize) -> Self {
        self.max_threads = Some(threads.clamp(1, 64));
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub detected_gpu_hint: String,
    pub vulkan_available: bool,
    pub rocm_available: bool,
    pub recommended_training_device: String,
    #[serde(default)]
    pub vulkan_device_name: Option<String>,
    #[serde(default)]
    pub vulkan_vram_mb: Option<u64>,
    #[serde(default)]
    pub total_system_ram_mb: Option<u64>,
    #[serde(default)]
    pub available_system_ram_mb: Option<u64>,
    #[serde(default)]
    pub recommended_runtime_profile: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub total_params: usize,
    pub weights_f32_mb: f64,
    pub weights_fp16_mb: f64,
    pub weights_int8_mb: f64,
    pub weights_int4_mb: f64,
    pub adamw_states_mb: f64,
    pub activations_est_mb: f64,
    pub total_training_est_mb: f64,
    pub inference_kv_fp32_mb: f64,
    pub inference_kv_fp16_mb: f64,
    pub inference_kv_int8_mb: f64,
    pub fits_16gb_ram: bool,
    pub fits_8gb_ram_inference: bool,
    pub fits_8gb_vram_rx7600: bool,
}

/// Detecta perfil básico de hardware local usando APIs padrão e sondagem segura de binários.
pub fn probe_hardware() -> HardwareProfile {
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let vk_adapter = crate::neural::VulkanDevice::try_adapter_info();
    let vulkan_available = vk_adapter.is_some() || command_exists("vulkaninfo");
    let rocm_available = command_exists("rocminfo") || std::env::var("ROCM_PATH").is_ok();

    let (detected_gpu_hint, vulkan_device_name, vulkan_vram_mb) = if let Some(ref info) = vk_adapter {
        (
            format!(
                "{} ({} MB VRAM, Vulkan {})",
                info.device_name, info.device_local_vram_mb, info.api_version_str
            ),
            Some(info.device_name.clone()),
            Some(info.device_local_vram_mb),
        )
    } else if cfg!(target_os = "windows") && vulkan_available {
        (
            "GPU local compatível com Vulkan detectada".to_string(),
            None,
            None,
        )
    } else if rocm_available {
        ("Ambiente ROCm detectado".to_string(), None, None)
    } else {
        (
            "Sem acelerador GPU dedicado detectado".to_string(),
            None,
            None,
        )
    };

    let (total_system_ram_mb, available_system_ram_mb) = match probe_system_ram_mb() {
        Some((tot, avail)) => (Some(tot), Some(avail)),
        None => (None, None),
    };
    let auto_profile = match total_system_ram_mb {
        Some(tot) if tot <= LOW_END_SYSTEM_RAM_THRESHOLD_MB => RuntimeProfileKind::LowEnd8Gb,
        _ => RuntimeProfileKind::Standard,
    };

    let active_threads = crate::neural::ops::effective_worker_threads();
    let recommended_training_device = if let Some(ref info) = vk_adapter {
        if info.is_discrete_gpu {
            format!(
                "Híbrido Auto: Vulkan GPU ({}) + CPU ({} threads, {})",
                info.device_name,
                active_threads,
                crate::neural::ops::simd_backend_name()
            )
        } else {
            format!(
                "CPU Multi-Thread ({} threads, {})",
                active_threads,
                crate::neural::ops::simd_backend_name()
            )
        }
    } else {
        format!(
            "CPU Multi-Thread ({} threads, {})",
            active_threads,
            crate::neural::ops::simd_backend_name()
        )
    };

    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus,
        detected_gpu_hint,
        vulkan_available,
        rocm_available,
        recommended_training_device,
        vulkan_device_name,
        vulkan_vram_mb,
        total_system_ram_mb,
        available_system_ram_mb,
        recommended_runtime_profile: Some(auto_profile.as_str().to_string()),
    }
}

/// Estima parâmetros e consumo real de memória (pesos f32 + grads + AdamW m/v + LayerCache + temporários)
/// de um Transformer Decoder-only, incluindo indicadores para 8 GB RAM (inferência) e 8 GB VRAM (RX 7600).
#[cfg_attr(not(test), allow(dead_code))]
pub fn estimate_transformer_memory(
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_layers: usize,
    d_ff: usize,
    batch_size: usize,
) -> MemoryEstimate {
    let inferred_heads = (d_model / 64).max(2);
    estimate_transformer_memory_with_heads(
        vocab_size,
        max_seq_len,
        d_model,
        inferred_heads,
        n_layers,
        d_ff,
        batch_size,
    )
}

/// Variante exata de [`estimate_transformer_memory`] parametrizada também por `n_heads`.
pub fn estimate_transformer_memory_with_heads(
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    batch_size: usize,
) -> MemoryEstimate {
    let t = max_seq_len;
    let d = d_model;
    let v = vocab_size;
    let h = n_heads.max(1);
    let b = batch_size.max(1);

    let embed_params = v * d + t * d;
    // Por camada (idêntico a TransformerConfig::total_params):
    // Q, K, V, O (4*d^2) + W1, W2 (2*d*d_ff) + b1 (d_ff) + b2 (d) + 2 RMSNorm (2*d)
    let per_layer_params = 4 * d * d + 2 * d * d_ff + d_ff + 3 * d;
    let lm_head_params = d + d * v;
    let total_params = embed_params + n_layers * per_layer_params + lm_head_params;

    let bytes_per_f32 = 4.0;
    let mb = 1024.0 * 1024.0;
    let weights_f32_mb = (total_params as f64 * bytes_per_f32) / mb;

    // Parâmetros mantidos em FP32 durante quantização: `pos_emb` + `L * (2*d + d_ff + d)` + `ln_f`
    let kept_f32_params = t * d + n_layers * (3 * d + d_ff) + d;
    let quant_2d_params = total_params.saturating_sub(kept_f32_params);
    let kept_f32_bytes = kept_f32_params as f64 * 4.0;
    let int8_rows = v + n_layers * (5 * d + d_ff) + d;
    let int4_blocks = (quant_2d_params + crate::neural::quant::INT4_GROUP_SIZE - 1)
        / crate::neural::quant::INT4_GROUP_SIZE;

    let weights_fp16_mb = (kept_f32_bytes + quant_2d_params as f64 * 2.0) / mb;
    let weights_int8_mb = (kept_f32_bytes + quant_2d_params as f64 + int8_rows as f64 * 4.0) / mb;
    let weights_int4_mb =
        (kept_f32_bytes + (quant_2d_params as f64 * 0.5) + int4_blocks as f64 * 2.0) / mb;

    // Grads (1x) + AdamW m (1x) + AdamW v (1x) = 3x pesos
    let adamw_states_mb = weights_f32_mb * 3.0;

    // Fórmula auditada (Agente 5): LayerCache vivo nas L camadas + temporários (logits + d_logits + d_*)
    let cache_per_sample = n_layers * (8 * t * d + 2 * t * d_ff + 2 * t + h * t * t);
    let temp_per_sample = 2 * t * v + 14 * t * d + 2 * t * d_ff + t;
    let act_elements = b * (cache_per_sample + temp_per_sample);
    let activations_est_mb = (act_elements as f64 * bytes_per_f32) / mb;
    let total_training_est_mb = weights_f32_mb + adamw_states_mb + activations_est_mb;

    // Inferência com KV Cache (`2 * L * T * d`) + scratch de decodificação de 1 token (`v + 4*d + d_ff`)
    let kv_only_elems = 2 * n_layers * t * d;
    let decode_scratch_mb = ((v + 4 * d + d_ff) as f64 * bytes_per_f32) / mb;
    let kv_fp32_mb = (kv_only_elems as f64 * 4.0) / mb;
    let kv_fp16_mb = (kv_only_elems as f64 * 2.0) / mb;
    let kv_int8_mb = (kv_only_elems as f64 + (2 * n_layers * t * 4) as f64) / mb;

    let inference_kv_fp32_mb = weights_f32_mb + kv_fp32_mb + decode_scratch_mb;
    let inference_kv_fp16_mb = weights_fp16_mb + kv_fp16_mb + decode_scratch_mb;
    let inference_kv_int8_mb = weights_int8_mb + kv_int8_mb + decode_scratch_mb;

    MemoryEstimate {
        total_params,
        weights_f32_mb,
        weights_fp16_mb,
        weights_int8_mb,
        weights_int4_mb,
        adamw_states_mb,
        activations_est_mb,
        total_training_est_mb,
        inference_kv_fp32_mb,
        inference_kv_fp16_mb,
        inference_kv_int8_mb,
        fits_16gb_ram: total_training_est_mb < 10_240.0,
        fits_8gb_ram_inference: inference_kv_fp32_mb < 4_096.0,
        fits_8gb_vram_rx7600: total_training_est_mb < 6_800.0,
    }
}

/// Micro-benchmark prático de multiplicação de matrizes na CPU local usando o kernel ativo
/// do motor neural (`ops::matmul_nn`: `CpuThreadPool` + tiling + AVX2/FMA quando disponível).
/// Retorna `(latência_por_chamada_ms, gflops_efetivos)`.
pub fn benchmark_cpu_matmul(size: usize) -> (u128, f64) {
    let n = size.max(32);
    let a = vec![0.5f32; n * n];
    let b = vec![0.25f32; n * n];
    let mut c = vec![0.0f32; n * n];

    // Warmup único do pool persistente para não misturar custo de criação inicial da thread
    crate::neural::ops::matmul_nn(&a, &b, &mut c, n, n, n);
    c.fill(0.0);

    let iters: usize = if n <= 256 { 32 } else { 8 };
    let start = Instant::now();
    for _ in 0..iters {
        crate::neural::ops::matmul_nn(&a, &b, &mut c, n, n, n);
    }
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let total_flops = 2.0 * (iters as f64) * (n as f64).powi(3);
    let gflops = (total_flops / secs) / 1e9;
    let avg_ms = ((secs * 1000.0) / iters as f64).round() as u128;
    std::hint::black_box(&c);
    (avg_ms, gflops)
}

/// Micro-benchmark de multiplicação matricial FP32 residente na VRAM da GPU Vulkan local (`size × size`).
/// Retorna `Some((latencia_media_ms, gflops_vram))` se o backend Vulkan Compute estiver ativo.
pub fn benchmark_gpu_matmul(size: usize) -> Option<(f64, f64)> {
    let dev = crate::neural::VulkanDevice::global().ok()?;
    dev.benchmark_vram_matmul(size).ok()
}

/// Sondagem nativa `100% std` da RAM física total e disponível do sistema operacional em MiB.
/// - Windows: `GlobalMemoryStatusEx` (`kernel32.dll`, já vinculada pelo runtime padrão do Rust);
/// - Linux: leitura de `MemTotal:` e `MemAvailable:` em `/proc/meminfo`.
pub fn probe_system_ram_mb() -> Option<(u64, u64)> {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }

        extern "system" {
            fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
        }

        let mut mem = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        // SAFETY: `mem` está propriamente inicializado com `dw_length = sizeof(MemoryStatusEx)` na stack.
        let ok = unsafe { GlobalMemoryStatusEx(&mut mem) };
        if ok != 0 && mem.ull_total_phys > 0 {
            let mb = 1024 * 1024;
            return Some((mem.ull_total_phys / mb, mem.ull_avail_phys / mb));
        }
        None
    }

    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut total_kb: Option<u64> = None;
        let mut avail_kb: Option<u64> = None;
        let mut free_kb: Option<u64> = None;
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<u64>().ok());
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                avail_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<u64>().ok());
            } else if let Some(rest) = line.strip_prefix("MemFree:") {
                free_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<u64>().ok());
            }
        }
        let tot = total_kb?;
        let av = avail_kb.or(free_kb).unwrap_or(tot / 2);
        Some((tot / 1024, av / 1024))
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
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
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            let tot = hw.total_system_ram_mb.expect("RAM total deve ser detectada");
            let avail = hw
                .available_system_ram_mb
                .expect("RAM livre deve ser detectada");
            assert!(tot >= 512 && avail > 0 && avail <= tot);
        }
        let low = super::RuntimeBudget::low_end_8gb();
        assert_eq!(low.profile, super::RuntimeProfileKind::LowEnd8Gb);
        assert_eq!(
            low.default_kv_precision,
            crate::neural::KvCachePrecision::Fp16
        );
        assert_eq!(
            low.max_inference_ram_bytes,
            Some(super::DEFAULT_LOW_END_INFERENCE_BUDGET_MB * 1024 * 1024)
        );
    }

    #[test]
    fn estimates_memory_for_small_and_medium_models() {
        let small = estimate_transformer_memory(2048, 128, 128, 4, 512, 4);
        assert!(small.total_params > 500_000);
        assert!(small.fits_16gb_ram);
        assert!(small.fits_8gb_ram_inference);
        assert!(small.fits_8gb_vram_rx7600);

        let mid_15m = super::estimate_transformer_memory_with_heads(4096, 256, 384, 6, 6, 1536, 1);
        let cfg_15m = crate::neural::TransformerConfig {
            vocab_size: 4096,
            max_seq_len: 256,
            d_model: 384,
            n_heads: 6,
            n_layers: 6,
            d_ff: 1536,
        };
        // Contagem exata de parâmetros deve coincidir 100% com `TransformerConfig::total_params`
        assert_eq!(mid_15m.total_params, cfg_15m.total_params());
        assert!(mid_15m.activations_est_mb > 55.0);
        assert!(mid_15m.fits_16gb_ram);
        assert!(mid_15m.fits_8gb_ram_inference);
        assert!(mid_15m.fits_8gb_vram_rx7600);
    }

    #[test]
    fn cpu_matmul_benchmark_runs() {
        let (_ms, gflops) = benchmark_cpu_matmul(64);
        assert!(gflops > 0.0);
    }
}
