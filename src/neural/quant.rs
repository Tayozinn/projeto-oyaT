//! Subsistema de Otimização de Memória e Quantização Pós-Treino (`SM-007` — Etapa 8).
//!
//! Implementa do zero em Rust puro (`100% std`):
//! 1. Quatro representações compactas de pesos para distribuição e inferência em PCs modestos:
//!    - **`Fp16`** (IEEE 754 binary16, `2` bytes/peso, *round-to-nearest-even* com saturação segura);
//!    - **`Bf16`** (Brain Float 16, `2` bytes/peso, *round-to-nearest-even*);
//!    - **`Int8`** (Simétrico por linha/canal `[-127, 127]` + `scale: f32`, `~1,02` byte/peso);
//!    - **`Int4`** (Simétrico em blocos de até `32` pesos `[-7, +7]` empacotados 2 por byte + `scale: FP16`, `~0,56` byte/peso).
//! 2. Inferência direta sem expansão global na RAM (`QuantizedModelData`):
//!    - Mantém `RMSNorm gamma`, `biases` e `pos_emb` em `FP32` (`< 0,5%` dos parâmetros) para preservar
//!      estabilidade numérica, enquanto todas as matrizes densas (`token_emb`, `wq`, `wk`, `wv`, `wo`,
//!      `w1`, `w2`, `lm_head`) permanecem comprimidas na RAM durante toda a execução;
//!    - Inclui kernels escalares de referência permanentes (Regra 3) e kernels acelerados em **AVX2 + FMA**.
//! 3. Formato binário versionado de distribuição **`SMQ1`** com escrita atômica e checksum **CRC32**.

use super::ops::{effective_worker_threads, has_avx2_fma, parallel_disjoint_rows_mut, worker_threads};
use super::{
    atomic_rename_with_retry, make_atomic_temp_path, Crc32Hasher, ServerMindTransformer,
    TransformerConfig,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const SMQ_MAGIC: &[u8; 4] = b"SMQ1";
const SMQ_VERSION: u32 = 1;
const MAX_SMQ_META_BYTES: usize = 1_048_576; // 1 MiB
const SMQ_IO_BUF_CAP: usize = 256 * 1024;
pub const INT4_GROUP_SIZE: usize = 32;

/// Precisão de armazenamento dos pesos de inferência.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightPrecision {
    Fp16,
    Bf16,
    Int8,
    Int4,
}

impl WeightPrecision {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fp16" | "f16" => Ok(Self::Fp16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "int8" | "q8" | "q8_0" => Ok(Self::Int8),
            "int4" | "q4" | "q4_0" => Ok(Self::Int4),
            other => Err(format!(
                "Formato de quantização inválido '{}': use 'fp16', 'bf16', 'int8' ou 'int4'.",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

/// Precisão configurável das chaves e valores armazenados no `KvCache`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCachePrecision {
    Fp32,
    Fp16,
    Int8,
}

impl Default for KvCachePrecision {
    fn default() -> Self {
        Self::Fp32
    }
}

impl KvCachePrecision {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::Fp32),
            "fp16" | "f16" => Ok(Self::Fp16),
            "int8" | "q8" => Ok(Self::Int8),
            other => Err(format!(
                "Precisão de KV Cache inválida '{}': use 'fp32', 'fp16' ou 'int8'.",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }
}

// ===========================================================================
// Conversores Escalares Exatos IEEE-754 `FP16` e `BF16`
// ===========================================================================

/// Converte `f32` para IEEE-754 `binary16` (`u16`) com arredondamento *round-to-nearest-even*
/// e saturação finita segura em `±65504.0` (sem produzir `NaN` ou `Inf`).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    if val.is_nan() {
        return 0;
    }
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs_bits = bits & 0x7FFF_FFFF;
    if abs_bits == 0 {
        return sign;
    }

    let exp32 = ((abs_bits >> 23) & 0xFF) as i32;
    let mant32 = abs_bits & 0x007F_FFFF;
    if exp32 == 0xFF {
        // Infinito satura no maior FP16 finito (±65504.0 = 0x7BFF)
        return sign | 0x7BFF;
    }

    let mut exp16 = exp32 - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7BFF;
    }

    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let m = mant32 | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let half_bit = 1u32 << (shift - 1);
        let rem_mask = half_bit - 1;
        let lsb = (m >> shift) & 1;
        let round_up = (m & half_bit) != 0 && ((m & rem_mask) != 0 || lsb != 0);
        let rounded = (m >> shift) + u32::from(round_up);
        return sign | (rounded as u16);
    }

    let half_bit = 0x1000u32;
    let rem_mask = 0x0FFFu32;
    let lsb = (mant32 >> 13) & 1;
    let round_up = (mant32 & half_bit) != 0 && ((mant32 & rem_mask) != 0 || lsb != 0);
    let mut mant16 = (mant32 >> 13) + u32::from(round_up);
    if mant16 == 0x0400 {
        mant16 = 0;
        exp16 += 1;
        if exp16 >= 31 {
            return sign | 0x7BFF;
        }
    }
    sign | ((exp16 as u16) << 10) | (mant16 as u16)
}

/// Converte IEEE-754 `binary16` (`u16`) de volta para `f32` exato.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: `(-1)^s * mant * 2^-24`
        let mag = (mant as f32) * (1.0f32 / 16_777_216.0f32);
        return if sign != 0 { -mag } else { mag };
    }
    if exp == 31 {
        let out_bits = sign | 0x7F7F_FFFF;
        return f32::from_bits(out_bits);
    }

    let exp32 = exp + (127 - 15);
    f32::from_bits(sign | (exp32 << 23) | (mant << 13))
}

/// Converte `f32` para `BF16` (`u16`) com *round-to-nearest-even* e saturação finita.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    if val.is_nan() {
        return 0;
    }
    if val == f32::INFINITY {
        return 0x7F7F;
    }
    if val == f32::NEG_INFINITY {
        return 0xFF7F;
    }
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    let bias = 0x7FFFu32 + lsb;
    let rounded = (bits.wrapping_add(bias) >> 16) as u16;
    if (rounded & 0x7FFF) >= 0x7F80 {
        (rounded & 0x8000) | 0x7F7F
    } else {
        rounded
    }
}

/// Converte `BF16` (`u16`) para `f32` exato.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ===========================================================================
// Matriz Quantizada (`QuantMatrix`) e Operações de Referência / AVX2
// ===========================================================================

#[derive(Debug, Clone)]
pub enum QuantMatrix {
    Fp16 {
        rows: usize,
        cols: usize,
        data: Vec<u16>,
    },
    Bf16 {
        rows: usize,
        cols: usize,
        data: Vec<u16>,
    },
    Int8 {
        rows: usize,
        cols: usize,
        scales: Vec<f32>,
        quants: Vec<i8>,
    },
    Int4 {
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
        packed_bytes_per_row: usize,
        /// Escalas em `FP16` (`rows * blocks_per_row`).
        scales_f16: Vec<u16>,
        /// Nibbles empacotados (`rows * packed_bytes_per_row`): nibble par nos 4 bits baixos,
        /// nibble ímpar nos 4 bits altos, codificados como `(q + 8) as u8` com `q in [-7, +7]`.
        packed: Vec<u8>,
    },
}

impl QuantMatrix {
    pub fn from_f32(src: &[f32], rows: usize, cols: usize, prec: WeightPrecision) -> Self {
        assert_eq!(src.len(), rows * cols);
        match prec {
            WeightPrecision::Fp16 => Self::Fp16 {
                rows,
                cols,
                data: src.iter().map(|&v| f32_to_f16(v)).collect(),
            },
            WeightPrecision::Bf16 => Self::Bf16 {
                rows,
                cols,
                data: src.iter().map(|&v| f32_to_bf16(v)).collect(),
            },
            WeightPrecision::Int8 => {
                let mut scales = Vec::with_capacity(rows);
                let mut quants = vec![0i8; rows * cols];
                for r in 0..rows {
                    let row = &src[r * cols..(r + 1) * cols];
                    let max_abs = row
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
                    scales.push(scale);
                    if scale > 0.0 {
                        let inv = 1.0 / scale;
                        let q_row = &mut quants[r * cols..(r + 1) * cols];
                        for c in 0..cols {
                            let v = if row[c].is_finite() { row[c] } else { 0.0 };
                            q_row[c] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                Self::Int8 {
                    rows,
                    cols,
                    scales,
                    quants,
                }
            }
            WeightPrecision::Int4 => {
                let blocks_per_row = (cols + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
                let packed_bytes_per_row = (cols + 1) / 2;
                let mut scales_f16 = vec![0u16; rows * blocks_per_row];
                let mut packed = vec![0x88u8; rows * packed_bytes_per_row]; // 0x88 => (0+8, 0+8) = (0, 0)

                for r in 0..rows {
                    let row = &src[r * cols..(r + 1) * cols];
                    let p_row = &mut packed[r * packed_bytes_per_row..(r + 1) * packed_bytes_per_row];
                    for b in 0..blocks_per_row {
                        let c_start = b * INT4_GROUP_SIZE;
                        let c_end = (c_start + INT4_GROUP_SIZE).min(cols);
                        let seg = &row[c_start..c_end];
                        let max_abs = seg
                            .iter()
                            .copied()
                            .filter(|v| v.is_finite())
                            .fold(0.0f32, |m, v| m.max(v.abs()));
                        let raw_scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
                        let sc_f16 = f32_to_f16(raw_scale);
                        let eff_scale = f16_to_f32(sc_f16);
                        scales_f16[r * blocks_per_row + b] = sc_f16;

                        if eff_scale > 0.0 {
                            let inv = 1.0 / eff_scale;
                            for c in c_start..c_end {
                                let v = if row[c].is_finite() { row[c] } else { 0.0 };
                                let q = (v * inv).round().clamp(-7.0, 7.0) as i8;
                                let nib = (q + 8) as u8;
                                let byte_idx = c / 2;
                                if c % 2 == 0 {
                                    p_row[byte_idx] = (p_row[byte_idx] & 0xF0) | (nib & 0x0F);
                                } else {
                                    p_row[byte_idx] = (p_row[byte_idx] & 0x0F) | ((nib & 0x0F) << 4);
                                }
                            }
                        }
                    }
                }

                Self::Int4 {
                    rows,
                    cols,
                    blocks_per_row,
                    packed_bytes_per_row,
                    scales_f16,
                    packed,
                }
            }
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(dead_code)]
    pub fn precision(&self) -> WeightPrecision {
        match self {
            Self::Fp16 { .. } => WeightPrecision::Fp16,
            Self::Bf16 { .. } => WeightPrecision::Bf16,
            Self::Int8 { .. } => WeightPrecision::Int8,
            Self::Int4 { .. } => WeightPrecision::Int4,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Fp16 { data, .. } | Self::Bf16 { data, .. } => data.len() * 2,
            Self::Int8 { scales, quants, .. } => scales.len() * 4 + quants.len(),
            Self::Int4 {
                scales_f16, packed, ..
            } => scales_f16.len() * 2 + packed.len(),
        }
    }

    /// Extrai e desquantiza uma única linha `row_idx` (usado para lookup $O(d)$ em `token_emb`).
    pub fn dequantize_row_into(&self, row_idx: usize, out: &mut [f32]) {
        match self {
            Self::Fp16 { rows, cols, data } => {
                assert!(row_idx < *rows && out.len() >= *cols);
                let r = &data[row_idx * cols..(row_idx + 1) * cols];
                for c in 0..*cols {
                    out[c] = f16_to_f32(r[c]);
                }
            }
            Self::Bf16 { rows, cols, data } => {
                assert!(row_idx < *rows && out.len() >= *cols);
                let r = &data[row_idx * cols..(row_idx + 1) * cols];
                for c in 0..*cols {
                    out[c] = bf16_to_f32(r[c]);
                }
            }
            Self::Int8 {
                rows,
                cols,
                scales,
                quants,
            } => {
                assert!(row_idx < *rows && out.len() >= *cols);
                let sc = scales[row_idx];
                let r = &quants[row_idx * cols..(row_idx + 1) * cols];
                for c in 0..*cols {
                    out[c] = (r[c] as f32) * sc;
                }
            }
            Self::Int4 {
                rows,
                cols,
                blocks_per_row,
                packed_bytes_per_row,
                scales_f16,
                packed,
            } => {
                assert!(row_idx < *rows && out.len() >= *cols);
                let p_row = &packed[row_idx * packed_bytes_per_row..(row_idx + 1) * packed_bytes_per_row];
                let sc_row = &scales_f16[row_idx * blocks_per_row..(row_idx + 1) * blocks_per_row];
                for b in 0..*blocks_per_row {
                    let sc = f16_to_f32(sc_row[b]);
                    let c_start = b * INT4_GROUP_SIZE;
                    let c_end = (c_start + INT4_GROUP_SIZE).min(*cols);
                    for c in c_start..c_end {
                        let byte = p_row[c / 2];
                        let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        let q = (nib as i32) - 8;
                        out[c] = (q as f32) * sc;
                    }
                }
            }
        }
    }

    /// Referência escalar permanente (Regra 3): `c[m, n] += a[m, k] @ W_quant[k, n]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn matmul_nn_reference(&self, a: &[f32], c: &mut [f32], m: usize) {
        let (k, n) = self.shape();
        assert!(a.len() >= m * k && c.len() >= m * n);
        let mut w_row = vec![0.0f32; n];
        for kk in 0..k {
            self.dequantize_row_into(kk, &mut w_row);
            for i in 0..m {
                let av = a[i * k + kk];
                if av == 0.0 {
                    continue;
                }
                let c_row = &mut c[i * n..(i + 1) * n];
                for j in 0..n {
                    c_row[j] += av * w_row[j];
                }
            }
        }
    }

    /// Multiplicação otimizada `c[m, n] += a[m, k] @ W_quant[k, n]` descompactando apenas
    /// em registradores/L1 cache sem alocar matrizes `FP32` na RAM, com paralelismo multi-thread
    /// automático sobre as linhas de entrada (`m >= 2`) durante o prefill (`SM-008`).
    pub fn matmul_nn(&self, a: &[f32], c: &mut [f32], m: usize) {
        let (k, n) = self.shape();
        if m == 0 || k == 0 || n == 0 {
            return;
        }
        assert!(a.len() >= m * k && c.len() >= m * n);
        let use_avx = has_avx2_fma();

        let work = m.saturating_mul(k).saturating_mul(n);
        if m >= 2 && work >= 131_072 && effective_worker_threads() > 1 {
            parallel_disjoint_rows_mut(&mut c[..m * n], m, n, 1, |row_start, rows, c_sub| {
                for r in 0..rows {
                    let i = row_start + r;
                    let a_row = &a[i * k..(i + 1) * k];
                    let c_row = &mut c_sub[r * n..(r + 1) * n];
                    self.gemv_row_accum(a_row, c_row, use_avx);
                }
            });
            return;
        }

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];
            self.gemv_row_accum(a_row, c_row, use_avx);
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Fp16 { rows, cols, .. }
            | Self::Bf16 { rows, cols, .. }
            | Self::Int8 { rows, cols, .. }
            | Self::Int4 { rows, cols, .. } => (*rows, *cols),
        }
    }

    #[inline]
    fn gemv_row_accum(&self, a_row: &[f32], c_row: &mut [f32], use_avx: bool) {
        match self {
            Self::Fp16 { rows, cols, data } => {
                #[cfg(target_arch = "x86_64")]
                if use_avx {
                    unsafe { gemv_fp16_avx2(a_row, data, c_row, *rows, *cols) };
                    return;
                }
                let _ = use_avx;
                for kk in 0..*rows {
                    let av = a_row[kk];
                    if av == 0.0 {
                        continue;
                    }
                    let wr = &data[kk * cols..(kk + 1) * cols];
                    for j in 0..*cols {
                        c_row[j] += av * f16_to_f32(wr[j]);
                    }
                }
            }
            Self::Bf16 { rows, cols, data } => {
                #[cfg(target_arch = "x86_64")]
                if use_avx {
                    unsafe { gemv_bf16_avx2(a_row, data, c_row, *rows, *cols) };
                    return;
                }
                let _ = use_avx;
                for kk in 0..*rows {
                    let av = a_row[kk];
                    if av == 0.0 {
                        continue;
                    }
                    let wr = &data[kk * cols..(kk + 1) * cols];
                    for j in 0..*cols {
                        c_row[j] += av * bf16_to_f32(wr[j]);
                    }
                }
            }
            Self::Int8 {
                rows,
                cols,
                scales,
                quants,
            } => {
                #[cfg(target_arch = "x86_64")]
                if use_avx {
                    unsafe { gemv_int8_avx2(a_row, scales, quants, c_row, *rows, *cols) };
                    return;
                }
                let _ = use_avx;
                for kk in 0..*rows {
                    let alpha = a_row[kk] * scales[kk];
                    if alpha == 0.0 {
                        continue;
                    }
                    let qr = &quants[kk * cols..(kk + 1) * cols];
                    for j in 0..*cols {
                        c_row[j] += alpha * (qr[j] as f32);
                    }
                }
            }
            Self::Int4 {
                rows,
                cols,
                blocks_per_row,
                packed_bytes_per_row,
                scales_f16,
                packed,
            } => {
                #[cfg(target_arch = "x86_64")]
                if use_avx {
                    unsafe {
                        gemv_int4_avx2(
                            a_row,
                            scales_f16,
                            packed,
                            c_row,
                            *rows,
                            *cols,
                            *blocks_per_row,
                            *packed_bytes_per_row,
                        )
                    };
                    return;
                }
                let _ = use_avx;
                for kk in 0..*rows {
                    let av = a_row[kk];
                    if av == 0.0 {
                        continue;
                    }
                    let p_row = &packed[kk * packed_bytes_per_row..(kk + 1) * packed_bytes_per_row];
                    let sc_row = &scales_f16[kk * blocks_per_row..(kk + 1) * blocks_per_row];
                    for b in 0..*blocks_per_row {
                        let alpha = av * f16_to_f32(sc_row[b]);
                        if alpha == 0.0 {
                            continue;
                        }
                        let c_start = b * INT4_GROUP_SIZE;
                        let c_end = (c_start + INT4_GROUP_SIZE).min(*cols);
                        for j in c_start..c_end {
                            let byte = p_row[j / 2];
                            let nib = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                            c_row[j] += alpha * ((nib as i32 - 8) as f32);
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Micro-Kernels AVX2 + FMA de Desquantização On-The-Fly em Registradores
// ===========================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_bf16_avx2(a_row: &[f32], data: &[u16], c_row: &mut [f32], k: usize, n: usize) {
    let w_ptr = data.as_ptr();
    let c_ptr = c_row.as_mut_ptr();
    let _ = worker_threads();

    for kk in 0..k {
        let av = a_row[kk];
        if av == 0.0 {
            continue;
        }
        let va = _mm256_set1_ps(av);
        let wr = w_ptr.add(kk * n);
        let mut j = 0usize;
        while j + 16 <= n {
            let u0 = _mm_loadu_si128(wr.add(j) as *const __m128i);
            let u1 = _mm_loadu_si128(wr.add(j + 8) as *const __m128i);
            let f0 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(u0)));
            let f1 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(u1)));

            let c0 = _mm256_loadu_ps(c_ptr.add(j));
            let c1 = _mm256_loadu_ps(c_ptr.add(j + 8));
            _mm256_storeu_ps(c_ptr.add(j), _mm256_fmadd_ps(va, f0, c0));
            _mm256_storeu_ps(c_ptr.add(j + 8), _mm256_fmadd_ps(va, f1, c1));
            j += 16;
        }
        while j < n {
            *c_ptr.add(j) += av * bf16_to_f32(*wr.add(j));
            j += 1;
        }
    }
}

/// Decodifica 8 valores IEEE-754 `FP16` normais/zero/subnormais em registradores AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_f16x8_avx2(h8: __m128i) -> __m256 {
    let w = _mm256_cvtepu16_epi32(h8);
    let sign = _mm256_slli_epi32::<16>(_mm256_and_si256(w, _mm256_set1_epi32(0x8000)));
    let exp = _mm256_and_si256(_mm256_srli_epi32::<10>(w), _mm256_set1_epi32(0x1F));
    let mant = _mm256_and_si256(w, _mm256_set1_epi32(0x03FF));

    let is_zero_exp = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    let norm_bits = _mm256_or_si256(
        sign,
        _mm256_or_si256(
            _mm256_slli_epi32::<23>(_mm256_add_epi32(exp, _mm256_set1_epi32(112))),
            _mm256_slli_epi32::<13>(mant),
        ),
    );
    let norm_f = _mm256_castsi256_ps(norm_bits);

    // Caminho subnormal/zero: `mant * 2^-24` com sinal preservado
    let sub_mag = _mm256_mul_ps(_mm256_cvtepi32_ps(mant), _mm256_set1_ps(1.0 / 16_777_216.0));
    let sub_f = _mm256_or_ps(sub_mag, _mm256_castsi256_ps(sign));

    _mm256_blendv_ps(norm_f, sub_f, _mm256_castsi256_ps(is_zero_exp))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_fp16_avx2(a_row: &[f32], data: &[u16], c_row: &mut [f32], k: usize, n: usize) {
    let w_ptr = data.as_ptr();
    let c_ptr = c_row.as_mut_ptr();

    for kk in 0..k {
        let av = a_row[kk];
        if av == 0.0 {
            continue;
        }
        let va = _mm256_set1_ps(av);
        let wr = w_ptr.add(kk * n);
        let mut j = 0usize;
        while j + 16 <= n {
            let h0 = _mm_loadu_si128(wr.add(j) as *const __m128i);
            let h1 = _mm_loadu_si128(wr.add(j + 8) as *const __m128i);
            let f0 = decode_f16x8_avx2(h0);
            let f1 = decode_f16x8_avx2(h1);

            let c0 = _mm256_loadu_ps(c_ptr.add(j));
            let c1 = _mm256_loadu_ps(c_ptr.add(j + 8));
            _mm256_storeu_ps(c_ptr.add(j), _mm256_fmadd_ps(va, f0, c0));
            _mm256_storeu_ps(c_ptr.add(j + 8), _mm256_fmadd_ps(va, f1, c1));
            j += 16;
        }
        while j < n {
            *c_ptr.add(j) += av * f16_to_f32(*wr.add(j));
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_int8_avx2(
    a_row: &[f32],
    scales: &[f32],
    quants: &[i8],
    c_row: &mut [f32],
    k: usize,
    n: usize,
) {
    let q_ptr = quants.as_ptr();
    let c_ptr = c_row.as_mut_ptr();

    for kk in 0..k {
        let alpha = a_row[kk] * scales[kk];
        if alpha == 0.0 {
            continue;
        }
        let va = _mm256_set1_ps(alpha);
        let qr = q_ptr.add(kk * n);
        let mut j = 0usize;
        while j + 32 <= n {
            let q0 = _mm_loadl_epi64(qr.add(j) as *const __m128i);
            let q1 = _mm_loadl_epi64(qr.add(j + 8) as *const __m128i);
            let q2 = _mm_loadl_epi64(qr.add(j + 16) as *const __m128i);
            let q3 = _mm_loadl_epi64(qr.add(j + 24) as *const __m128i);

            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3));

            _mm256_storeu_ps(
                c_ptr.add(j),
                _mm256_fmadd_ps(va, f0, _mm256_loadu_ps(c_ptr.add(j))),
            );
            _mm256_storeu_ps(
                c_ptr.add(j + 8),
                _mm256_fmadd_ps(va, f1, _mm256_loadu_ps(c_ptr.add(j + 8))),
            );
            _mm256_storeu_ps(
                c_ptr.add(j + 16),
                _mm256_fmadd_ps(va, f2, _mm256_loadu_ps(c_ptr.add(j + 16))),
            );
            _mm256_storeu_ps(
                c_ptr.add(j + 24),
                _mm256_fmadd_ps(va, f3, _mm256_loadu_ps(c_ptr.add(j + 24))),
            );
            j += 32;
        }
        while j + 8 <= n {
            let q0 = _mm_loadl_epi64(qr.add(j) as *const __m128i);
            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
            _mm256_storeu_ps(
                c_ptr.add(j),
                _mm256_fmadd_ps(va, f0, _mm256_loadu_ps(c_ptr.add(j))),
            );
            j += 8;
        }
        while j < n {
            *c_ptr.add(j) += alpha * (*qr.add(j) as f32);
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_int4_avx2(
    a_row: &[f32],
    scales_f16: &[u16],
    packed: &[u8],
    c_row: &mut [f32],
    k: usize,
    n: usize,
    blocks_per_row: usize,
    packed_bytes_per_row: usize,
) {
    let p_ptr = packed.as_ptr();
    let c_ptr = c_row.as_mut_ptr();
    let low_mask = _mm_set1_epi8(0x0F);
    let bias8 = _mm_set1_epi8(8);

    for kk in 0..k {
        let av = a_row[kk];
        if av == 0.0 {
            continue;
        }
        let pr = p_ptr.add(kk * packed_bytes_per_row);
        let sr = &scales_f16[kk * blocks_per_row..(kk + 1) * blocks_per_row];

        for b in 0..blocks_per_row {
            let alpha = av * f16_to_f32(sr[b]);
            if alpha == 0.0 {
                continue;
            }
            let c_start = b * INT4_GROUP_SIZE;
            if c_start + 32 <= n {
                // Desempacota 16 bytes -> 32 valores i8 intercalados `(even, odd)` em 2 registradores XMM
                let raw16 = _mm_loadu_si128(pr.add(b * 16) as *const __m128i);
                let even = _mm_sub_epi8(_mm_and_si128(raw16, low_mask), bias8);
                let odd = _mm_sub_epi8(
                    _mm_and_si128(_mm_srli_epi16::<4>(raw16), low_mask),
                    bias8,
                );
                // Intercala bytes `even` e `odd` restaurando a ordem exata `c_start .. c_start + 32`
                let lo16 = _mm_unpacklo_epi8(even, odd); // elementos 0..15
                let hi16 = _mm_unpackhi_epi8(even, odd); // elementos 16..31

                let va = _mm256_set1_ps(alpha);
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo16)));
                let f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                let f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi16)));

                _mm256_storeu_ps(
                    c_ptr.add(c_start),
                    _mm256_fmadd_ps(va, f0, _mm256_loadu_ps(c_ptr.add(c_start))),
                );
                _mm256_storeu_ps(
                    c_ptr.add(c_start + 8),
                    _mm256_fmadd_ps(va, f1, _mm256_loadu_ps(c_ptr.add(c_start + 8))),
                );
                _mm256_storeu_ps(
                    c_ptr.add(c_start + 16),
                    _mm256_fmadd_ps(va, f2, _mm256_loadu_ps(c_ptr.add(c_start + 16))),
                );
                _mm256_storeu_ps(
                    c_ptr.add(c_start + 24),
                    _mm256_fmadd_ps(va, f3, _mm256_loadu_ps(c_ptr.add(c_start + 24))),
                );
            } else {
                let c_end = (c_start + INT4_GROUP_SIZE).min(n);
                for j in c_start..c_end {
                    let byte = *pr.add(j / 2);
                    let nib = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    *c_ptr.add(j) += alpha * ((nib as i32 - 8) as f32);
                }
            }
        }
    }
}

// ===========================================================================
// Estrutura de Pesos Quantizados do Transformer e Serialização `SMQ1`
// ===========================================================================

#[derive(Debug, Clone)]
pub struct QuantizedBlock {
    pub wq: QuantMatrix,
    pub wk: QuantMatrix,
    pub wv: QuantMatrix,
    pub wo: QuantMatrix,
    pub w1: QuantMatrix,
    pub w2: QuantMatrix,
}

#[derive(Debug, Clone)]
pub struct QuantizedModelData {
    pub precision: WeightPrecision,
    pub token_emb: QuantMatrix,
    pub blocks: Vec<QuantizedBlock>,
    pub lm_head: QuantMatrix,
}

impl QuantizedModelData {
    pub fn from_fp32_model(model: &ServerMindTransformer, prec: WeightPrecision) -> Result<Self, String> {
        if model.is_quantized() {
            return Err("O modelo de origem já está quantizado; forneça um checkpoint FP32.".to_string());
        }
        let d = model.config.d_model;
        let ff = model.config.d_ff;
        let v = model.config.vocab_size;

        let token_emb = QuantMatrix::from_f32(&model.token_emb, v, d, prec);
        let blocks = model
            .blocks
            .iter()
            .map(|b| QuantizedBlock {
                wq: QuantMatrix::from_f32(&b.wq, d, d, prec),
                wk: QuantMatrix::from_f32(&b.wk, d, d, prec),
                wv: QuantMatrix::from_f32(&b.wv, d, d, prec),
                wo: QuantMatrix::from_f32(&b.wo, d, d, prec),
                w1: QuantMatrix::from_f32(&b.w1, d, ff, prec),
                w2: QuantMatrix::from_f32(&b.w2, ff, d, prec),
            })
            .collect();
        let lm_head = QuantMatrix::from_f32(&model.lm_head, d, v, prec);

        Ok(Self {
            precision: prec,
            token_emb,
            blocks,
            lm_head,
        })
    }

    pub fn memory_bytes(&self) -> usize {
        let blks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.wq.memory_bytes()
                    + b.wk.memory_bytes()
                    + b.wv.memory_bytes()
                    + b.wo.memory_bytes()
                    + b.w1.memory_bytes()
                    + b.w2.memory_bytes()
            })
            .sum();
        self.token_emb.memory_bytes() + blks + self.lm_head.memory_bytes()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedCheckpointMetadata {
    pub config: TransformerConfig,
    pub precision: WeightPrecision,
    pub step: usize,
    #[serde(with = "super::serde_loss_f32")]
    pub best_val_loss: f32,
}

/// Converte um `ServerMindTransformer` `FP32` em um `ServerMindTransformer` quantizado em memória
/// (liberando todos os tensores densos 2D em `FP32` para consumo mínimo de RAM).
pub fn quantize_model_in_memory(
    model: &ServerMindTransformer,
    precision: WeightPrecision,
) -> Result<ServerMindTransformer, String> {
    let qdata = QuantizedModelData::from_fp32_model(model, precision)?;
    let blocks = model
        .blocks
        .iter()
        .map(|b| super::TransformerBlock {
            ln1_gamma: b.ln1_gamma.clone(),
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            ln2_gamma: b.ln2_gamma.clone(),
            w1: Vec::new(),
            b1: b.b1.clone(),
            w2: Vec::new(),
            b2: b.b2.clone(),
        })
        .collect();

    Ok(ServerMindTransformer {
        token_emb: Vec::new(),
        pos_emb: model.pos_emb.clone(),
        blocks,
        ln_f_gamma: model.ln_f_gamma.clone(),
        lm_head: Vec::new(),
        config: model.config.clone(),
        quantized: Some(Box::new(qdata)),
    })
}

/// Salva um modelo quantizado no formato binário compacto `SMQ1` com escrita atômica e trailer CRC32.
pub fn save_quantized_checkpoint(
    path: &Path,
    qmodel: &ServerMindTransformer,
    step: usize,
    best_val_loss: f32,
) -> Result<(), String> {
    let Some(ref qdata) = qmodel.quantized else {
        return Err("save_quantized_checkpoint requer um modelo quantizado.".to_string());
    };
    qmodel.config.validate()?;

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => {
            fs::create_dir_all(p)
                .map_err(|e| format!("Falha ao criar pasta {:?}: {}", p, e))?;
            p
        }
        _ => Path::new("."),
    };

    let meta = QuantizedCheckpointMetadata {
        config: qmodel.config.clone(),
        precision: qdata.precision,
        step,
        best_val_loss,
    };
    let meta_bytes = serde_json::to_vec(&meta)
        .map_err(|e| format!("Falha ao serializar metadados SMQ1: {}", e))?;
    if meta_bytes.len() > MAX_SMQ_META_BYTES {
        return Err("Metadados SMQ1 excedem limite máximo.".to_string());
    }

    let tmp_path = make_atomic_temp_path(parent, path);
    let write_res = (|| -> Result<(), String> {
        let file = File::create(&tmp_path)
            .map_err(|e| format!("Falha ao criar arquivo temporário {:?}: {}", tmp_path, e))?;
        let mut w = BufWriter::with_capacity(SMQ_IO_BUF_CAP, file);
        let mut crc = Crc32Hasher::new();

        write_bytes_crc(&mut w, &mut crc, SMQ_MAGIC)?;
        write_bytes_crc(&mut w, &mut crc, &SMQ_VERSION.to_le_bytes())?;
        write_bytes_crc(&mut w, &mut crc, &(meta_bytes.len() as u32).to_le_bytes())?;
        write_bytes_crc(&mut w, &mut crc, &meta_bytes)?;

        // 1. Escreve tensores FP32 pequenos preservados (`pos_emb`, `ln1/ln2/b1/b2`, `ln_f_gamma`)
        write_f32_slice_crc(&mut w, &mut crc, &qmodel.pos_emb)?;
        for b in &qmodel.blocks {
            write_f32_slice_crc(&mut w, &mut crc, &b.ln1_gamma)?;
            write_f32_slice_crc(&mut w, &mut crc, &b.ln2_gamma)?;
            write_f32_slice_crc(&mut w, &mut crc, &b.b1)?;
            write_f32_slice_crc(&mut w, &mut crc, &b.b2)?;
        }
        write_f32_slice_crc(&mut w, &mut crc, &qmodel.ln_f_gamma)?;

        // 2. Escreve matrizes quantizadas
        write_quant_matrix_crc(&mut w, &mut crc, &qdata.token_emb)?;
        for qb in &qdata.blocks {
            write_quant_matrix_crc(&mut w, &mut crc, &qb.wq)?;
            write_quant_matrix_crc(&mut w, &mut crc, &qb.wk)?;
            write_quant_matrix_crc(&mut w, &mut crc, &qb.wv)?;
            write_quant_matrix_crc(&mut w, &mut crc, &qb.wo)?;
            write_quant_matrix_crc(&mut w, &mut crc, &qb.w1)?;
            write_quant_matrix_crc(&mut w, &mut crc, &qb.w2)?;
        }
        write_quant_matrix_crc(&mut w, &mut crc, &qdata.lm_head)?;

        let checksum = crc.finalize();
        w.write_all(&checksum.to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;
        let inner = w.into_inner().map_err(|e| e.error().to_string())?;
        inner.sync_all().map_err(|e| e.to_string())?;
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    atomic_rename_with_retry(&tmp_path, path)
}

/// Carrega um pacote quantizado `SMQ1` diretamente em representação compacta na RAM,
/// verificando tamanho e checksum `CRC32` antes de expor o modelo.
pub fn load_quantized_checkpoint(
    path: &Path,
) -> Result<(ServerMindTransformer, QuantizedCheckpointMetadata), String> {
    let file = File::open(path)
        .map_err(|e| format!("Falha ao abrir modelo quantizado {:?}: {}", path, e))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("Falha ao ler metadados de {:?}: {}", path, e))?
        .len();
    if file_len < 16 {
        return Err("Arquivo SMQ1 truncado no cabeçalho.".to_string());
    }

    let mut r = BufReader::with_capacity(SMQ_IO_BUF_CAP, file);
    let mut crc = Crc32Hasher::new();

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != SMQ_MAGIC {
        return Err("Formato inválido: assinatura 'SMQ1' não encontrada.".to_string());
    }
    crc.update(&magic);

    let mut u32_buf = [0u8; 4];
    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let ver = u32::from_le_bytes(u32_buf);
    if ver != SMQ_VERSION {
        return Err(format!("Versão SMQ1 não suportada: {}", ver));
    }
    crc.update(&u32_buf);

    r.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let meta_len = u32::from_le_bytes(u32_buf) as usize;
    if meta_len == 0 || meta_len > MAX_SMQ_META_BYTES || (16 + meta_len as u64) > file_len {
        return Err("Tamanho de metadados inválido em arquivo SMQ1.".to_string());
    }
    crc.update(&u32_buf);

    let mut meta_bytes = vec![0u8; meta_len];
    r.read_exact(&mut meta_bytes).map_err(|e| e.to_string())?;
    crc.update(&meta_bytes);

    let meta: QuantizedCheckpointMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| format!("Metadados SMQ1 inválidos: {}", e))?;
    meta.config.validate()?;
    if meta.config.checked_total_params().is_none() {
        return Err("Overflow aritmético nas dimensões do modelo SMQ1.".to_string());
    }

    let cfg = &meta.config;
    let d = cfg.d_model;
    let ff = cfg.d_ff;
    let v = cfg.vocab_size;
    let prec = meta.precision;

    let mut pos_emb = vec![0.0f32; cfg.max_seq_len * d];
    read_f32_slice_crc(&mut r, &mut crc, &mut pos_emb)?;

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        let mut ln1_gamma = vec![0.0f32; d];
        let mut ln2_gamma = vec![0.0f32; d];
        let mut b1 = vec![0.0f32; ff];
        let mut b2 = vec![0.0f32; d];
        read_f32_slice_crc(&mut r, &mut crc, &mut ln1_gamma)?;
        read_f32_slice_crc(&mut r, &mut crc, &mut ln2_gamma)?;
        read_f32_slice_crc(&mut r, &mut crc, &mut b1)?;
        read_f32_slice_crc(&mut r, &mut crc, &mut b2)?;
        blocks.push(super::TransformerBlock {
            ln1_gamma,
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            ln2_gamma,
            w1: Vec::new(),
            b1,
            w2: Vec::new(),
            b2,
        });
    }

    let mut ln_f_gamma = vec![0.0f32; d];
    read_f32_slice_crc(&mut r, &mut crc, &mut ln_f_gamma)?;

    let token_emb = read_quant_matrix_crc(&mut r, &mut crc, v, d, prec)?;
    let mut qblocks = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        qblocks.push(QuantizedBlock {
            wq: read_quant_matrix_crc(&mut r, &mut crc, d, d, prec)?,
            wk: read_quant_matrix_crc(&mut r, &mut crc, d, d, prec)?,
            wv: read_quant_matrix_crc(&mut r, &mut crc, d, d, prec)?,
            wo: read_quant_matrix_crc(&mut r, &mut crc, d, d, prec)?,
            w1: read_quant_matrix_crc(&mut r, &mut crc, d, ff, prec)?,
            w2: read_quant_matrix_crc(&mut r, &mut crc, ff, d, prec)?,
        });
    }
    let lm_head = read_quant_matrix_crc(&mut r, &mut crc, d, v, prec)?;

    let mut crc_buf = [0u8; 4];
    r.read_exact(&mut crc_buf)
        .map_err(|e| format!("Falha ao ler CRC32 do arquivo SMQ1: {}", e))?;
    let stored_crc = u32::from_le_bytes(crc_buf);
    let computed_crc = crc.finalize();
    if stored_crc != computed_crc {
        return Err(format!(
            "Falha de integridade em {:?}: CRC32 inválido (esperado {:08x}, calculado {:08x}).",
            path, stored_crc, computed_crc
        ));
    }

    // Garante que não há bytes excedentes no final do arquivo
    let mut extra = [0u8; 1];
    if r.read(&mut extra).map_err(|e| e.to_string())? != 0 {
        return Err("Arquivo SMQ1 corrompido: bytes excedentes após trailer CRC32.".to_string());
    }

    let model = ServerMindTransformer {
        token_emb: Vec::new(),
        pos_emb,
        blocks,
        ln_f_gamma,
        lm_head: Vec::new(),
        config: cfg.clone(),
        quantized: Some(Box::new(QuantizedModelData {
            precision: prec,
            token_emb,
            blocks: qblocks,
            lm_head,
        })),
    };
    Ok((model, meta))
}

fn write_bytes_crc<W: Write>(w: &mut W, crc: &mut Crc32Hasher, bytes: &[u8]) -> Result<(), String> {
    w.write_all(bytes).map_err(|e| e.to_string())?;
    crc.update(bytes);
    Ok(())
}

fn write_f32_slice_crc<W: Write>(
    w: &mut W,
    crc: &mut Crc32Hasher,
    slice: &[f32],
) -> Result<(), String> {
    let mut buf = [0u8; 4096];
    for chunk in slice.chunks(1024) {
        let nb = chunk.len() * 4;
        for (i, &v) in chunk.iter().enumerate() {
            if !v.is_finite() {
                return Err("Valor não-finito detectado ao salvar SMQ1.".to_string());
            }
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        write_bytes_crc(w, crc, &buf[..nb])?;
    }
    Ok(())
}

fn read_f32_slice_crc<R: Read>(
    r: &mut R,
    crc: &mut Crc32Hasher,
    dst: &mut [f32],
) -> Result<(), String> {
    let mut buf = [0u8; 4096];
    for chunk in dst.chunks_mut(1024) {
        let nb = chunk.len() * 4;
        r.read_exact(&mut buf[..nb]).map_err(|e| e.to_string())?;
        crc.update(&buf[..nb]);
        for (i, slot) in chunk.iter_mut().enumerate() {
            let v = f32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ]);
            if !v.is_finite() {
                return Err("Valor não-finito detectado ao carregar SMQ1.".to_string());
            }
            *slot = v;
        }
    }
    Ok(())
}

fn write_u16_slice_crc<W: Write>(
    w: &mut W,
    crc: &mut Crc32Hasher,
    slice: &[u16],
) -> Result<(), String> {
    let mut buf = [0u8; 4096];
    for chunk in slice.chunks(2048) {
        let nb = chunk.len() * 2;
        for (i, &v) in chunk.iter().enumerate() {
            buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        write_bytes_crc(w, crc, &buf[..nb])?;
    }
    Ok(())
}

fn read_u16_slice_crc<R: Read>(
    r: &mut R,
    crc: &mut Crc32Hasher,
    dst: &mut [u16],
) -> Result<(), String> {
    let mut buf = [0u8; 4096];
    for chunk in dst.chunks_mut(2048) {
        let nb = chunk.len() * 2;
        r.read_exact(&mut buf[..nb]).map_err(|e| e.to_string())?;
        crc.update(&buf[..nb]);
        for (i, slot) in chunk.iter_mut().enumerate() {
            *slot = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
    }
    Ok(())
}

fn write_quant_matrix_crc<W: Write>(
    w: &mut W,
    crc: &mut Crc32Hasher,
    qm: &QuantMatrix,
) -> Result<(), String> {
    match qm {
        QuantMatrix::Fp16 { data, .. } | QuantMatrix::Bf16 { data, .. } => {
            write_u16_slice_crc(w, crc, data)
        }
        QuantMatrix::Int8 { scales, quants, .. } => {
            write_f32_slice_crc(w, crc, scales)?;
            let raw_u8 =
                unsafe { std::slice::from_raw_parts(quants.as_ptr() as *const u8, quants.len()) };
            write_bytes_crc(w, crc, raw_u8)
        }
        QuantMatrix::Int4 {
            scales_f16, packed, ..
        } => {
            write_u16_slice_crc(w, crc, scales_f16)?;
            write_bytes_crc(w, crc, packed)
        }
    }
}

fn read_quant_matrix_crc<R: Read>(
    r: &mut R,
    crc: &mut Crc32Hasher,
    rows: usize,
    cols: usize,
    prec: WeightPrecision,
) -> Result<QuantMatrix, String> {
    match prec {
        WeightPrecision::Fp16 => {
            let mut data = vec![0u16; rows * cols];
            read_u16_slice_crc(r, crc, &mut data)?;
            Ok(QuantMatrix::Fp16 { rows, cols, data })
        }
        WeightPrecision::Bf16 => {
            let mut data = vec![0u16; rows * cols];
            read_u16_slice_crc(r, crc, &mut data)?;
            Ok(QuantMatrix::Bf16 { rows, cols, data })
        }
        WeightPrecision::Int8 => {
            let mut scales = vec![0.0f32; rows];
            read_f32_slice_crc(r, crc, &mut scales)?;
            let mut quants = vec![0i8; rows * cols];
            let raw_u8 = unsafe {
                std::slice::from_raw_parts_mut(quants.as_mut_ptr() as *mut u8, quants.len())
            };
            r.read_exact(raw_u8).map_err(|e| e.to_string())?;
            crc.update(raw_u8);
            Ok(QuantMatrix::Int8 {
                rows,
                cols,
                scales,
                quants,
            })
        }
        WeightPrecision::Int4 => {
            let blocks_per_row = (cols + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
            let packed_bytes_per_row = (cols + 1) / 2;
            let mut scales_f16 = vec![0u16; rows * blocks_per_row];
            read_u16_slice_crc(r, crc, &mut scales_f16)?;
            let mut packed = vec![0u8; rows * packed_bytes_per_row];
            r.read_exact(&mut packed).map_err(|e| e.to_string())?;
            crc.update(&packed);
            Ok(QuantMatrix::Int4 {
                rows,
                cols,
                blocks_per_row,
                packed_bytes_per_row,
                scales_f16,
                packed,
            })
        }
    }
}
