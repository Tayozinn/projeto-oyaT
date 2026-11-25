//! Fundação Computacional de CPU (`SM-005` — Etapa 6).
//!
//! Fornece:
//! 1. `CpuThreadPool` persistente em `std` (zero crates externas, zero `CreateThread` por MatMul,
//!    seguro sob concorrência e sob `panic` via `catch_unwind` + guarda RAII);
//! 2. Implementações de referência permanentes (`matmul_nn_reference`, `matmul_tn_reference`,
//!    `matmul_nt_reference` — Regra 3);
//! 3. Kernels matriciais (`matmul_nn`, `matmul_tn`, `matmul_nt`) e vetor-matriz (`m = 1` GEMV)
//!    com *cache blocking* L1/L2 e micro-kernels registrador-residentes **6×16 / 2×4 / 1×32**
//!    em **AVX2 + FMA** (`is_x86_feature_detected!`), com fallback escalar blocado 100% seguro;
//! 4. Primitivas vetoriais auxiliares (`dot_f32`, `axpy_f32`) para acelerar produtos internos
//!    contíguos na atenção multi-cabeça.
//!
//! ### Nota de Precisão Numérica (Regra 2)
//! Quando AVX2+FMA está disponível em runtime, `_mm256_fmadd_ps` executa `rn(a * b + c)` com um
//! único arredondamento IEEE-754 (mais preciso que `rn(rn(a * b) + c)` escalar) e reduz produtos
//! internos em múltiplos acumuladores YMM. Diferenças de última unidade na faixa `~1e-6..1e-5`
//! em relação à soma escalar estrita são esperadas e verificadas por testes automatizados.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

/// Limite dinâmico opcional de threads ativas (`0` = usar todas as threads do pool).
static MAX_ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Limiar de operações (`m * k * n` MACs) para despachar GEMM multi-thread no pool residente.
const GEMM_PARALLEL_THRESHOLD: usize = 131_072;
/// Limiar de operações (`k * n` MACs) para paralelizar GEMV (`m == 1`) por blocos de colunas.
const GEMV_PARALLEL_THRESHOLD: usize = 262_144;

/// Dimensões de *cache blocking* calibradas para L1D (32 KiB) e L2 privado (256 KiB):
/// - Painel `B[KC, NC]` = `128 * 256 * 4 = 128 KiB` (reside em L2 com folga para `A` e `C`).
const TILE_KC: usize = 128;
const TILE_NC: usize = 256;
const TILE_MC_TN: usize = 128;
const TILE_JC_NT: usize = 64;
const TILE_NC_NT: usize = 1024;

/// Retorna o número máximo de threads de trabalho instanciadas pelo pool.
/// Respeita a variável de ambiente opcional `SERVERMIND_THREADS` (`1..=64`).
#[inline]
pub fn worker_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        if let Ok(raw) = std::env::var("SERVERMIND_THREADS") {
            if let Ok(parsed) = raw.trim().parse::<usize>() {
                if (1..=64).contains(&parsed) {
                    return parsed;
                }
            }
        }
        thread::available_parallelism()
            .map(|p| p.get().clamp(1, 32))
            .unwrap_or(2)
    })
}

/// Define dinamicamente um teto de threads ativas por despacho no `CpuThreadPool`
/// (`None` ou `0` restaura todas as `worker_threads()` disponíveis).
#[inline]
pub fn set_max_active_threads(max_threads: Option<usize>) {
    match max_threads {
        Some(n) if n >= 1 => MAX_ACTIVE_THREADS.store(n.min(64), Ordering::Relaxed),
        _ => MAX_ACTIVE_THREADS.store(0, Ordering::Relaxed),
    }
}

/// Número efetivo de threads ativas considerando tanto o pool físico quanto `set_max_active_threads`.
#[inline]
pub fn effective_worker_threads() -> usize {
    let base = worker_threads();
    let cap = MAX_ACTIVE_THREADS.load(Ordering::Relaxed);
    if cap == 0 {
        base
    } else {
        base.min(cap).max(1)
    }
}

/// Indica se a CPU atual suporta simultaneamente instruções AVX2 e FMA3 em runtime.
#[inline]
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS_SIMD: OnceLock<bool> = OnceLock::new();
        *HAS_SIMD.get_or_init(|| {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Nome legível do backend de instruções ativo na CPU local.
pub fn simd_backend_name() -> &'static str {
    if has_avx2_fma() {
        "AVX2+FMA"
    } else {
        "Scalar-Tiled"
    }
}

// ===========================================================================
// Pool de Threads Persistente Fork-Join (`CpuThreadPool`)
// ===========================================================================

fn noop_task(_task_idx: usize, _ctx_ptr: usize) {}

struct PoolState {
    generation: u64,
    num_bg_tasks: usize,
    remaining: usize,
    task_fn: fn(usize, usize),
    ctx_ptr: usize,
    worker_panicked: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct CpuThreadPool {
    total_threads: usize,
    dispatch_lock: Mutex<()>,
    shared: Arc<PoolShared>,
    _workers: Vec<JoinHandle<()>>,
}

struct BgCompletionGuard<'a> {
    shared: &'a PoolShared,
}

impl<'a> Drop for BgCompletionGuard<'a> {
    fn drop(&mut self) {
        let mut st = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if thread::panicking() {
            st.worker_panicked = true;
        }
        st.remaining = st.remaining.saturating_sub(1);
        if st.remaining == 0 {
            self.shared.done_cv.notify_all();
        }
    }
}

impl CpuThreadPool {
    fn global() -> &'static Self {
        static POOL: OnceLock<CpuThreadPool> = OnceLock::new();
        POOL.get_or_init(|| Self::new(worker_threads()))
    }

    fn new(total_threads: usize) -> Self {
        let total = total_threads.max(1);
        let bg_count = total.saturating_sub(1);
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState {
                generation: 0,
                num_bg_tasks: 0,
                remaining: 0,
                task_fn: noop_task,
                ctx_ptr: 0,
                worker_panicked: false,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(bg_count);
        for bg_idx in 0..bg_count {
            let shared_cl = Arc::clone(&shared);
            let builder = thread::Builder::new().name(format!("sm-cpu-w{}", bg_idx + 1));
            if let Ok(handle) = builder.spawn(move || {
                Self::worker_loop(&shared_cl, bg_idx);
            }) {
                handles.push(handle);
            }
        }

        CpuThreadPool {
            total_threads: 1 + handles.len(),
            dispatch_lock: Mutex::new(()),
            shared,
            _workers: handles,
        }
    }

    fn worker_loop(shared: &PoolShared, bg_idx: usize) {
        let mut seen_gen = 0u64;
        loop {
            let (task_fn, ctx_ptr, active) = {
                let mut st = shared
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                while st.generation == seen_gen {
                    st = shared
                        .work_cv
                        .wait(st)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                seen_gen = st.generation;
                if bg_idx < st.num_bg_tasks {
                    (st.task_fn, st.ctx_ptr, true)
                } else {
                    (noop_task as fn(usize, usize), 0usize, false)
                }
            };

            if active {
                let _guard = BgCompletionGuard { shared };
                let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                    task_fn(bg_idx + 1, ctx_ptr);
                }));
            }
        }
    }

    /// Executa `num_tasks` em paralelo (`0` na thread chamadora e `1..num_tasks` nos workers residentes).
    /// Garante barreira completa antes de retornar mesmo em caso de panic.
    fn execute<F>(&self, requested_tasks: usize, body: F)
    where
        F: Fn(usize) + Sync,
    {
        let num_tasks = requested_tasks.min(self.total_threads);
        if num_tasks <= 1 {
            body(0);
            return;
        }

        let _dispatch_guard = self
            .dispatch_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        fn trampoline<F: Fn(usize) + Sync>(task_idx: usize, ctx_ptr: usize) {
            // SAFETY: `ctx_ptr` aponta para `body` vivo na stack da thread chamadora até
            // `st.remaining == 0` ser confirmado abaixo (inclusive sob `catch_unwind`).
            let f = unsafe { &*(ctx_ptr as *const F) };
            f(task_idx);
        }

        let bg_tasks = num_tasks - 1;
        {
            let mut st = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            st.generation = st.generation.wrapping_add(1);
            st.num_bg_tasks = bg_tasks;
            st.remaining = bg_tasks;
            st.task_fn = trampoline::<F>;
            st.ctx_ptr = (&body as *const F) as usize;
            st.worker_panicked = false;
        }

        // Acorda todos os workers residentes para que `bg_idx < bg_tasks` assumam suas fatias imediatamente.
        self.shared.work_cv.notify_all();

        let caller_res = panic::catch_unwind(AssertUnwindSafe(|| {
            body(0);
        }));

        let worker_panicked = {
            let mut st = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            while st.remaining > 0 {
                st = self
                    .shared
                    .done_cv
                    .wait(st)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            st.ctx_ptr = 0;
            st.task_fn = noop_task;
            st.worker_panicked
        };

        if let Err(payload) = caller_res {
            panic::resume_unwind(payload);
        }
        assert!(
            !worker_panicked,
            "Worker do CpuThreadPool entrou em panic durante operação matricial."
        );
    }
}

/// Divide `total_rows` linhas de largura `row_width` em fatias mutáveis estritamente disjuntas
/// de `c` e executa `body(row_start, rows, c_sub)` em paralelo no `CpuThreadPool`.
pub(super) fn parallel_disjoint_rows_mut<F>(
    c: &mut [f32],
    total_rows: usize,
    row_width: usize,
    align_rows: usize,
    body: F,
) where
    F: Fn(usize, usize, &mut [f32]) + Sync,
{
    assert!(row_width > 0 && align_rows > 0);
    let total_elems = total_rows
        .checked_mul(row_width)
        .expect("overflow em parallel_disjoint_rows_mut");
    assert!(c.len() >= total_elems);
    if total_rows == 0 {
        return;
    }

    let pool = CpuThreadPool::global();
    let cap = MAX_ACTIVE_THREADS.load(Ordering::Relaxed);
    let max_t = if cap == 0 {
        pool.total_threads
    } else {
        pool.total_threads.min(cap).max(1)
    };
    if max_t <= 1 || total_rows <= align_rows {
        body(0, total_rows, &mut c[..total_elems]);
        return;
    }

    let raw_chunk = (total_rows + max_t - 1) / max_t;
    let chunk_rows = ((raw_chunk + align_rows - 1) / align_rows) * align_rows;
    let num_tasks = (total_rows + chunk_rows - 1) / chunk_rows;
    if num_tasks <= 1 {
        body(0, total_rows, &mut c[..total_elems]);
        return;
    }

    let c_addr = c.as_mut_ptr() as usize;
    pool.execute(num_tasks, |task_idx| {
        let row_start = task_idx * chunk_rows;
        if row_start >= total_rows {
            return;
        }
        let rows = (total_rows - row_start).min(chunk_rows);
        let elem_offset = row_start * row_width;
        let elem_len = rows * row_width;
        // SAFETY: Para cada `task_idx` distinto em `0..num_tasks`, os intervalos
        // `[row_start, row_start + rows)` são estritamente disjuntos e contidos em
        // `0..total_rows`, com `total_rows * row_width <= c.len()`. A barreira de `execute`
        // aguarda todos os workers antes de liberar o empréstimo exclusivo `&mut [f32]`.
        let c_sub =
            unsafe { std::slice::from_raw_parts_mut((c_addr as *mut f32).add(elem_offset), elem_len) };
        body(row_start, rows, c_sub);
    });
}

// ===========================================================================
// Implementações de Referência Permanentes (Regra 3)
// ===========================================================================

/// Referência escalar simples: `C[m, n] += A[m, k] @ B[k, n]`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn matmul_nn_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for (kk, &av) in a_row.iter().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// Referência escalar simples: `C[k, n] += A[m, k]^T @ B[m, n]`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn matmul_tn_reference(a: &[f32], b: &[f32], c: &mut [f32], k: usize, m: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= m * n && c.len() >= k * n);
    for row_m in 0..m {
        let a_row = &a[row_m * k..(row_m + 1) * k];
        let b_row = &b[row_m * n..(row_m + 1) * n];
        for (i, &av) in a_row.iter().enumerate() {
            let c_row = &mut c[i * n..(i + 1) * n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *cv += av * bv;
            }
        }
    }
}

/// Referência escalar simples: `C[m, k] += A[m, n] @ B[k, n]^T`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn matmul_nt_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * n && b.len() >= k * n && c.len() >= m * k);
    for i in 0..m {
        let a_row = &a[i * n..(i + 1) * n];
        let c_row = &mut c[i * k..(i + 1) * k];
        for j in 0..k {
            let b_row = &b[j * n..(j + 1) * n];
            let mut sum = 0.0f32;
            for (&av, &bv) in a_row.iter().zip(b_row.iter()) {
                sum += av * bv;
            }
            c_row[j] += sum;
        }
    }
}

// ===========================================================================
// Primitivas Vetoriais (`dot_f32` e `axpy_f32`)
// ===========================================================================

/// Produto interno contíguo `sum_i a[i] * b[i]` com aceleração AVX2+FMA quando disponível.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 16 && has_avx2_fma() {
            // SAFETY: `has_avx2_fma()` verificado em runtime e `len <= a.len(), b.len()`.
            return unsafe { dot_avx2_fma(a.as_ptr(), b.as_ptr(), len) };
        }
    }
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut i = 0usize;
    while i + 2 <= len {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        i += 2;
    }
    if i < len {
        s0 += a[i] * b[i];
    }
    s0 + s1
}

/// Acumulação linear contígua `y[i] += alpha * x[i]` com aceleração AVX2+FMA quando disponível.
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let len = x.len().min(y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 16 && has_avx2_fma() {
            // SAFETY: `has_avx2_fma()` verificado em runtime e `len <= x.len(), y.len()`.
            unsafe { axpy_avx2_fma(alpha, x.as_ptr(), y.as_mut_ptr(), len) };
            return;
        }
    }
    for i in 0..len {
        y[i] += alpha * x[i];
    }
}

// ===========================================================================
// Entradas Públicas Otimizadas (`matmul_nn`, `matmul_tn`, `matmul_nt`)
// ===========================================================================

/// `C[m, n] += A[m, k] @ B[k, n]` com *cache blocking*, AVX2+FMA e `CpuThreadPool`.
pub fn matmul_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);

    let use_avx = has_avx2_fma();

    // Caminho especializado GEMV (`m == 1`, ex.: decodificação autoregressiva com `KvCache`).
    if m == 1 {
        let work = k * n;
        if work >= GEMV_PARALLEL_THRESHOLD && n >= 64 && effective_worker_threads() > 1 {
            // Tratamos blocos de 32 colunas de `c[0..n]` como "linhas" de largura 32 para particionar sem alias.
            let full_blocks = n / 32;
            let main_cols = full_blocks * 32;
            if full_blocks >= 2 {
                parallel_disjoint_rows_mut(&mut c[..main_cols], full_blocks, 32, 1, |blk_start, blk_cnt, c_sub| {
                    let col_start = blk_start * 32;
                    let cols = blk_cnt * 32;
                    gemv_col_slice(&a[..k], b, c_sub, k, n, col_start, cols, use_avx);
                });
                if main_cols < n {
                    let rem = n - main_cols;
                    gemv_col_slice(
                        &a[..k],
                        b,
                        &mut c[main_cols..n],
                        k,
                        n,
                        main_cols,
                        rem,
                        use_avx,
                    );
                }
                return;
            }
        }
        gemv_col_slice(&a[..k], b, &mut c[..n], k, n, 0, n, use_avx);
        return;
    }

    let work = m * k * n;
    if work >= GEMM_PARALLEL_THRESHOLD && m >= 4 && effective_worker_threads() > 1 {
        parallel_disjoint_rows_mut(c, m, n, 6, |row_start, rows, c_sub| {
            let a_sub = &a[row_start * k..(row_start + rows) * k];
            matmul_nn_single(a_sub, b, c_sub, rows, k, n, use_avx);
        });
    } else {
        matmul_nn_single(&a[..m * k], &b[..k * n], &mut c[..m * n], m, k, n, use_avx);
    }
}

/// `C[k, n] += A[m, k]^T @ B[m, n]` com *cache blocking* L2, AVX2+FMA e `CpuThreadPool`.
pub fn matmul_tn(a: &[f32], b: &[f32], c: &mut [f32], k: usize, m: usize, n: usize) {
    if k == 0 || m == 0 || n == 0 {
        return;
    }
    assert!(a.len() >= m * k && b.len() >= m * n && c.len() >= k * n);

    let use_avx = has_avx2_fma();
    let work = k * m * n;
    if work >= GEMM_PARALLEL_THRESHOLD && k >= 6 && effective_worker_threads() > 1 {
        parallel_disjoint_rows_mut(c, k, n, 6, |row_start, rows, c_sub| {
            matmul_tn_row_slice(a, b, c_sub, k, m, n, row_start, rows, use_avx);
        });
    } else {
        matmul_tn_row_slice(a, b, &mut c[..k * n], k, m, n, 0, k, use_avx);
    }
}

/// `C[m, k] += A[m, n] @ B[k, n]^T` com micro-kernel 2×4 AVX2+FMA e `CpuThreadPool`.
pub fn matmul_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(a.len() >= m * n && b.len() >= k * n && c.len() >= m * k);

    let use_avx = has_avx2_fma();
    let work = m * n * k;
    if work >= GEMM_PARALLEL_THRESHOLD && m >= 4 && effective_worker_threads() > 1 {
        parallel_disjoint_rows_mut(c, m, k, 2, |row_start, rows, c_sub| {
            let a_sub = &a[row_start * n..(row_start + rows) * n];
            matmul_nt_single(a_sub, b, c_sub, rows, n, k, use_avx);
        });
    } else {
        matmul_nt_single(&a[..m * n], &b[..k * n], &mut c[..m * k], m, n, k, use_avx);
    }
}

// ===========================================================================
// Despachantes Single-Slice (Tiled Scalar vs. Tiled AVX2+FMA)
// ===========================================================================

#[inline]
fn gemv_col_slice(
    a_row: &[f32],
    b: &[f32],
    c_out: &mut [f32],
    k: usize,
    n_stride: usize,
    col_start: usize,
    cols: usize,
    use_avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx {
            // SAFETY: `has_avx2_fma()` confirmado em `use_avx` e limites validados pelo chamador.
            unsafe {
                gemv_col_slice_avx2(a_row, b, c_out, k, n_stride, col_start, cols);
            }
            return;
        }
    }
    let _ = use_avx;
    for jc in (0..cols).step_by(TILE_NC) {
        let jc_len = (cols - jc).min(TILE_NC);
        for kc in (0..k).step_by(TILE_KC) {
            let kc_end = (kc + TILE_KC).min(k);
            for kk in kc..kc_end {
                let av = a_row[kk];
                let b_sub = &b[kk * n_stride + col_start + jc..kk * n_stride + col_start + jc + jc_len];
                let c_sub = &mut c_out[jc..jc + jc_len];
                for (cv, &bv) in c_sub.iter_mut().zip(b_sub.iter()) {
                    *cv += av * bv;
                }
            }
        }
    }
}

#[inline]
fn matmul_nn_single(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    use_avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx {
            // SAFETY: `has_avx2_fma()` confirmado e fatias verificadas.
            unsafe {
                matmul_nn_avx2(a, b, c, m, k, n);
            }
            return;
        }
    }
    let _ = use_avx;
    for jc in (0..n).step_by(TILE_NC) {
        let jc_end = (jc + TILE_NC).min(n);
        for kc in (0..k).step_by(TILE_KC) {
            let kc_end = (kc + TILE_KC).min(k);
            for i in 0..m {
                let a_row = &a[i * k + kc..i * k + kc_end];
                let c_row = &mut c[i * n + jc..i * n + jc_end];
                for (off_k, &av) in a_row.iter().enumerate() {
                    let kk = kc + off_k;
                    let b_row = &b[kk * n + jc..kk * n + jc_end];
                    for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                        *cv += av * bv;
                    }
                }
            }
        }
    }
}

#[inline]
fn matmul_tn_row_slice(
    a: &[f32],
    b: &[f32],
    c_sub: &mut [f32],
    k_total: usize,
    m: usize,
    n: usize,
    row_start: usize,
    rows: usize,
    use_avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx {
            // SAFETY: `has_avx2_fma()` confirmado e fatias verificadas.
            unsafe {
                matmul_tn_avx2(a, b, c_sub, k_total, m, n, row_start, rows);
            }
            return;
        }
    }
    let _ = use_avx;
    for jc in (0..n).step_by(TILE_NC) {
        let jc_end = (jc + TILE_NC).min(n);
        for mc in (0..m).step_by(TILE_MC_TN) {
            let mc_end = (mc + TILE_MC_TN).min(m);
            for row_m in mc..mc_end {
                let a_sub = &a[row_m * k_total + row_start..row_m * k_total + row_start + rows];
                let b_sub = &b[row_m * n + jc..row_m * n + jc_end];
                for (i, &av) in a_sub.iter().enumerate() {
                    let c_row = &mut c_sub[i * n + jc..i * n + jc_end];
                    for (cv, &bv) in c_row.iter_mut().zip(b_sub.iter()) {
                        *cv += av * bv;
                    }
                }
            }
        }
    }
}

#[inline]
fn matmul_nt_single(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    use_avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx {
            // SAFETY: `has_avx2_fma()` confirmado e fatias verificadas.
            unsafe {
                matmul_nt_avx2(a, b, c, m, n, k);
            }
            return;
        }
    }
    let _ = use_avx;
    for nc in (0..n).step_by(TILE_NC_NT) {
        let nc_end = (nc + TILE_NC_NT).min(n);
        for jc in (0..k).step_by(TILE_JC_NT) {
            let jc_end = (jc + TILE_JC_NT).min(k);
            for i in 0..m {
                let a_row = &a[i * n + nc..i * n + nc_end];
                let c_row = &mut c[i * k..i * k + k];
                for j in jc..jc_end {
                    let b_row = &b[j * n + nc..j * n + nc_end];
                    let mut sum = 0.0f32;
                    for (&av, &bv) in a_row.iter().zip(b_row.iter()) {
                        sum += av * bv;
                    }
                    c_row[j] += sum;
                }
            }
        }
    }
}

// ===========================================================================
// Micro-Kernels x86_64 AVX2 + FMA (`#[target_feature(enable = "avx2,fma")]`)
// ===========================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps::<0x55>(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= len {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 8)),
            _mm256_loadu_ps(b.add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 16)),
            _mm256_loadu_ps(b.add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(i + 24)),
            _mm256_loadu_ps(b.add(i + 24)),
            acc3,
        );
        i += 32;
    }

    while i + 8 <= len {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
        i += 8;
    }

    let sum01 = _mm256_add_ps(acc0, acc1);
    let sum23 = _mm256_add_ps(acc2, acc3);
    let mut total = hsum256_ps(_mm256_add_ps(sum01, sum23));

    while i < len {
        total += *a.add(i) * *b.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy_avx2_fma(alpha: f32, x: *const f32, y: *mut f32, len: usize) {
    let va = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 32 <= len {
        let y0 = _mm256_loadu_ps(y.add(i));
        let y1 = _mm256_loadu_ps(y.add(i + 8));
        let y2 = _mm256_loadu_ps(y.add(i + 16));
        let y3 = _mm256_loadu_ps(y.add(i + 24));
        _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i)), y0));
        _mm256_storeu_ps(
            y.add(i + 8),
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 8)), y1),
        );
        _mm256_storeu_ps(
            y.add(i + 16),
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 16)), y2),
        );
        _mm256_storeu_ps(
            y.add(i + 24),
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i + 24)), y3),
        );
        i += 32;
    }
    while i + 8 <= len {
        let y0 = _mm256_loadu_ps(y.add(i));
        _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i)), y0));
        i += 8;
    }
    while i < len {
        *y.add(i) += alpha * *x.add(i);
        i += 1;
    }
}

/// GEMV AVX2+FMA (`1 × cols`): mantém faixas de 32 floats (`4` registradores YMM) residentes
/// durante todo o bloco `kc..kc_end`, reduzindo tráfego de escrita em `C` em até `128×`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_col_slice_avx2(
    a_row: &[f32],
    b: &[f32],
    c_out: &mut [f32],
    k: usize,
    n_stride: usize,
    col_start: usize,
    cols: usize,
) {
    assert!(a_row.len() >= k && c_out.len() >= cols && b.len() >= (k - 1) * n_stride + col_start + cols);
    let a_ptr = a_row.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c_out.as_mut_ptr();

    for jc in (0..cols).step_by(TILE_NC) {
        let jc_end = (jc + TILE_NC).min(cols);
        for kc in (0..k).step_by(TILE_KC) {
            let kc_end = (kc + TILE_KC).min(k);
            let mut j = jc;
            while j + 32 <= jc_end {
                let c_off = c_ptr.add(j);
                let b_col = col_start + j;
                let mut c0 = _mm256_loadu_ps(c_off);
                let mut c1 = _mm256_loadu_ps(c_off.add(8));
                let mut c2 = _mm256_loadu_ps(c_off.add(16));
                let mut c3 = _mm256_loadu_ps(c_off.add(24));

                for kk in kc..kc_end {
                    let va = _mm256_set1_ps(*a_ptr.add(kk));
                    let br = b_ptr.add(kk * n_stride + b_col);
                    c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br), c0);
                    c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(8)), c1);
                    c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(16)), c2);
                    c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(24)), c3);
                }

                _mm256_storeu_ps(c_off, c0);
                _mm256_storeu_ps(c_off.add(8), c1);
                _mm256_storeu_ps(c_off.add(16), c2);
                _mm256_storeu_ps(c_off.add(24), c3);
                j += 32;
            }

            while j + 8 <= jc_end {
                let c_off = c_ptr.add(j);
                let b_col = col_start + j;
                let mut c0 = _mm256_loadu_ps(c_off);
                for kk in kc..kc_end {
                    let va = _mm256_set1_ps(*a_ptr.add(kk));
                    c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_ptr.add(kk * n_stride + b_col)), c0);
                }
                _mm256_storeu_ps(c_off, c0);
                j += 8;
            }

            if j < jc_end {
                for kk in kc..kc_end {
                    let av = *a_ptr.add(kk);
                    let br = b_ptr.add(kk * n_stride + col_start);
                    for col in j..jc_end {
                        *c_ptr.add(col) += av * *br.add(col);
                    }
                }
            }
        }
    }
}

/// GEMM `matmul_nn` AVX2+FMA com micro-kernel **6×16** registrador-residente (`12` acumuladores YMM).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_nn_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for jc in (0..n).step_by(TILE_NC) {
        let jc_end = (jc + TILE_NC).min(n);
        for kc in (0..k).step_by(TILE_KC) {
            let kc_end = (kc + TILE_KC).min(k);

            let mut i = 0usize;
            while i + 6 <= m {
                let mut j = jc;
                while j + 16 <= jc_end {
                    let mut c00 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j));
                    let mut c01 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j + 8));
                    let mut c10 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j));
                    let mut c11 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j + 8));
                    let mut c20 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j));
                    let mut c21 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j + 8));
                    let mut c30 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j));
                    let mut c31 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j + 8));
                    let mut c40 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j));
                    let mut c41 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j + 8));
                    let mut c50 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j));
                    let mut c51 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j + 8));

                    for kk in kc..kc_end {
                        let b0 = _mm256_loadu_ps(b_ptr.add(kk * n + j));
                        let b1 = _mm256_loadu_ps(b_ptr.add(kk * n + j + 8));

                        let a0 = _mm256_set1_ps(*a_ptr.add((i + 0) * k + kk));
                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a0, b1, c01);

                        let a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * k + kk));
                        c10 = _mm256_fmadd_ps(a1, b0, c10);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);

                        let a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * k + kk));
                        c20 = _mm256_fmadd_ps(a2, b0, c20);
                        c21 = _mm256_fmadd_ps(a2, b1, c21);

                        let a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * k + kk));
                        c30 = _mm256_fmadd_ps(a3, b0, c30);
                        c31 = _mm256_fmadd_ps(a3, b1, c31);

                        let a4 = _mm256_set1_ps(*a_ptr.add((i + 4) * k + kk));
                        c40 = _mm256_fmadd_ps(a4, b0, c40);
                        c41 = _mm256_fmadd_ps(a4, b1, c41);

                        let a5 = _mm256_set1_ps(*a_ptr.add((i + 5) * k + kk));
                        c50 = _mm256_fmadd_ps(a5, b0, c50);
                        c51 = _mm256_fmadd_ps(a5, b1, c51);
                    }

                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j), c00);
                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j + 8), c01);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), c10);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j + 8), c11);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j), c20);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j + 8), c21);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j), c30);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j + 8), c31);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j), c40);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j + 8), c41);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j), c50);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j + 8), c51);
                    j += 16;
                }

                if j + 8 <= jc_end {
                    let mut c0 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j));
                    let mut c1 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j));
                    let mut c2 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j));
                    let mut c3 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j));
                    let mut c4 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j));
                    let mut c5 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j));
                    for kk in kc..kc_end {
                        let b0 = _mm256_loadu_ps(b_ptr.add(kk * n + j));
                        c0 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 0) * k + kk)), b0, c0);
                        c1 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 1) * k + kk)), b0, c1);
                        c2 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 2) * k + kk)), b0, c2);
                        c3 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 3) * k + kk)), b0, c3);
                        c4 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 4) * k + kk)), b0, c4);
                        c5 = _mm256_fmadd_ps(_mm256_set1_ps(*a_ptr.add((i + 5) * k + kk)), b0, c5);
                    }
                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j), c0);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), c1);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j), c2);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j), c3);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j), c4);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j), c5);
                    j += 8;
                }

                if j < jc_end {
                    for kk in kc..kc_end {
                        for r in 0..6 {
                            let av = *a_ptr.add((i + r) * k + kk);
                            for col in j..jc_end {
                                *c_ptr.add((i + r) * n + col) += av * *b_ptr.add(kk * n + col);
                            }
                        }
                    }
                }
                i += 6;
            }

            // Linhas residuais (`m % 6`: 1..=5 linhas) com registradores 1×32 / 1×8 + cauda escalar
            while i < m {
                let mut j = jc;
                while j + 32 <= jc_end {
                    let c_base = c_ptr.add(i * n + j);
                    let mut c0 = _mm256_loadu_ps(c_base);
                    let mut c1 = _mm256_loadu_ps(c_base.add(8));
                    let mut c2 = _mm256_loadu_ps(c_base.add(16));
                    let mut c3 = _mm256_loadu_ps(c_base.add(24));
                    for kk in kc..kc_end {
                        let va = _mm256_set1_ps(*a_ptr.add(i * k + kk));
                        let br = b_ptr.add(kk * n + j);
                        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br), c0);
                        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(8)), c1);
                        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(16)), c2);
                        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(24)), c3);
                    }
                    _mm256_storeu_ps(c_base, c0);
                    _mm256_storeu_ps(c_base.add(8), c1);
                    _mm256_storeu_ps(c_base.add(16), c2);
                    _mm256_storeu_ps(c_base.add(24), c3);
                    j += 32;
                }
                while j + 8 <= jc_end {
                    let c_base = c_ptr.add(i * n + j);
                    let mut c0 = _mm256_loadu_ps(c_base);
                    for kk in kc..kc_end {
                        let va = _mm256_set1_ps(*a_ptr.add(i * k + kk));
                        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_ptr.add(kk * n + j)), c0);
                    }
                    _mm256_storeu_ps(c_base, c0);
                    j += 8;
                }
                if j < jc_end {
                    for kk in kc..kc_end {
                        let av = *a_ptr.add(i * k + kk);
                        for col in j..jc_end {
                            *c_ptr.add(i * n + col) += av * *b_ptr.add(kk * n + col);
                        }
                    }
                }
                i += 1;
            }
        }
    }
}

/// GEMM `matmul_tn` (`C[k, n] += A[m, k]^T @ B[m, n]`) com micro-kernel **6×16** registrador-residente:
/// elimina o *cache thrashing* em `C` mantendo `6×16` saídas em YMM ao longo do tile `mc..mc_end`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_tn_avx2(
    a: &[f32],
    b: &[f32],
    c_sub: &mut [f32],
    k_total: usize,
    m: usize,
    n: usize,
    row_start: usize,
    rows: usize,
) {
    assert!(
        a.len() >= (m - 1) * k_total + row_start + rows
            && b.len() >= m * n
            && c_sub.len() >= rows * n
    );
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c_sub.as_mut_ptr();

    for jc in (0..n).step_by(TILE_NC) {
        let jc_end = (jc + TILE_NC).min(n);
        for mc in (0..m).step_by(TILE_MC_TN) {
            let mc_end = (mc + TILE_MC_TN).min(m);

            let mut i = 0usize;
            while i + 6 <= rows {
                let a_col_base = row_start + i;
                let mut j = jc;
                while j + 16 <= jc_end {
                    let mut c00 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j));
                    let mut c01 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j + 8));
                    let mut c10 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j));
                    let mut c11 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j + 8));
                    let mut c20 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j));
                    let mut c21 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j + 8));
                    let mut c30 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j));
                    let mut c31 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j + 8));
                    let mut c40 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j));
                    let mut c41 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j + 8));
                    let mut c50 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j));
                    let mut c51 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j + 8));

                    for row_m in mc..mc_end {
                        let b0 = _mm256_loadu_ps(b_ptr.add(row_m * n + j));
                        let b1 = _mm256_loadu_ps(b_ptr.add(row_m * n + j + 8));
                        let ar = a_ptr.add(row_m * k_total + a_col_base);

                        let a0 = _mm256_set1_ps(*ar);
                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a0, b1, c01);

                        let a1 = _mm256_set1_ps(*ar.add(1));
                        c10 = _mm256_fmadd_ps(a1, b0, c10);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);

                        let a2 = _mm256_set1_ps(*ar.add(2));
                        c20 = _mm256_fmadd_ps(a2, b0, c20);
                        c21 = _mm256_fmadd_ps(a2, b1, c21);

                        let a3 = _mm256_set1_ps(*ar.add(3));
                        c30 = _mm256_fmadd_ps(a3, b0, c30);
                        c31 = _mm256_fmadd_ps(a3, b1, c31);

                        let a4 = _mm256_set1_ps(*ar.add(4));
                        c40 = _mm256_fmadd_ps(a4, b0, c40);
                        c41 = _mm256_fmadd_ps(a4, b1, c41);

                        let a5 = _mm256_set1_ps(*ar.add(5));
                        c50 = _mm256_fmadd_ps(a5, b0, c50);
                        c51 = _mm256_fmadd_ps(a5, b1, c51);
                    }

                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j), c00);
                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j + 8), c01);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), c10);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j + 8), c11);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j), c20);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j + 8), c21);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j), c30);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j + 8), c31);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j), c40);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j + 8), c41);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j), c50);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j + 8), c51);
                    j += 16;
                }

                if j + 8 <= jc_end {
                    let mut c0 = _mm256_loadu_ps(c_ptr.add((i + 0) * n + j));
                    let mut c1 = _mm256_loadu_ps(c_ptr.add((i + 1) * n + j));
                    let mut c2 = _mm256_loadu_ps(c_ptr.add((i + 2) * n + j));
                    let mut c3 = _mm256_loadu_ps(c_ptr.add((i + 3) * n + j));
                    let mut c4 = _mm256_loadu_ps(c_ptr.add((i + 4) * n + j));
                    let mut c5 = _mm256_loadu_ps(c_ptr.add((i + 5) * n + j));
                    for row_m in mc..mc_end {
                        let b0 = _mm256_loadu_ps(b_ptr.add(row_m * n + j));
                        let ar = a_ptr.add(row_m * k_total + a_col_base);
                        c0 = _mm256_fmadd_ps(_mm256_set1_ps(*ar), b0, c0);
                        c1 = _mm256_fmadd_ps(_mm256_set1_ps(*ar.add(1)), b0, c1);
                        c2 = _mm256_fmadd_ps(_mm256_set1_ps(*ar.add(2)), b0, c2);
                        c3 = _mm256_fmadd_ps(_mm256_set1_ps(*ar.add(3)), b0, c3);
                        c4 = _mm256_fmadd_ps(_mm256_set1_ps(*ar.add(4)), b0, c4);
                        c5 = _mm256_fmadd_ps(_mm256_set1_ps(*ar.add(5)), b0, c5);
                    }
                    _mm256_storeu_ps(c_ptr.add((i + 0) * n + j), c0);
                    _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), c1);
                    _mm256_storeu_ps(c_ptr.add((i + 2) * n + j), c2);
                    _mm256_storeu_ps(c_ptr.add((i + 3) * n + j), c3);
                    _mm256_storeu_ps(c_ptr.add((i + 4) * n + j), c4);
                    _mm256_storeu_ps(c_ptr.add((i + 5) * n + j), c5);
                    j += 8;
                }

                if j < jc_end {
                    for row_m in mc..mc_end {
                        let ar = a_ptr.add(row_m * k_total + a_col_base);
                        for r in 0..6 {
                            let av = *ar.add(r);
                            for col in j..jc_end {
                                *c_ptr.add((i + r) * n + col) += av * *b_ptr.add(row_m * n + col);
                            }
                        }
                    }
                }
                i += 6;
            }

            // Linhas residuais (`rows % 6`)
            while i < rows {
                let a_col = row_start + i;
                let mut j = jc;
                while j + 32 <= jc_end {
                    let c_base = c_ptr.add(i * n + j);
                    let mut c0 = _mm256_loadu_ps(c_base);
                    let mut c1 = _mm256_loadu_ps(c_base.add(8));
                    let mut c2 = _mm256_loadu_ps(c_base.add(16));
                    let mut c3 = _mm256_loadu_ps(c_base.add(24));
                    for row_m in mc..mc_end {
                        let va = _mm256_set1_ps(*a_ptr.add(row_m * k_total + a_col));
                        let br = b_ptr.add(row_m * n + j);
                        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br), c0);
                        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(8)), c1);
                        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(16)), c2);
                        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(br.add(24)), c3);
                    }
                    _mm256_storeu_ps(c_base, c0);
                    _mm256_storeu_ps(c_base.add(8), c1);
                    _mm256_storeu_ps(c_base.add(16), c2);
                    _mm256_storeu_ps(c_base.add(24), c3);
                    j += 32;
                }
                while j + 8 <= jc_end {
                    let c_base = c_ptr.add(i * n + j);
                    let mut c0 = _mm256_loadu_ps(c_base);
                    for row_m in mc..mc_end {
                        let va = _mm256_set1_ps(*a_ptr.add(row_m * k_total + a_col));
                        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_ptr.add(row_m * n + j)), c0);
                    }
                    _mm256_storeu_ps(c_base, c0);
                    j += 8;
                }
                if j < jc_end {
                    for row_m in mc..mc_end {
                        let av = *a_ptr.add(row_m * k_total + a_col);
                        for col in j..jc_end {
                            *c_ptr.add(i * n + col) += av * *b_ptr.add(row_m * n + col);
                        }
                    }
                }
                i += 1;
            }
        }
    }
}

/// GEMM `matmul_nt` (`C[m, k] += A[m, n] @ B[k, n]^T`) com micro-kernel **2×4** AVX2+FMA:
/// rompe a cadeia de latência de soma escalar acumulando `8` produtos internos em paralelo.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_nt_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * n && b.len() >= k * n && c.len() >= m * k);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for nc in (0..n).step_by(TILE_NC_NT) {
        let nc_end = (nc + TILE_NC_NT).min(n);
        let seg_len = nc_end - nc;

        for jc in (0..k).step_by(TILE_JC_NT) {
            let jc_end = (jc + TILE_JC_NT).min(k);

            let mut i = 0usize;
            while i + 2 <= m {
                let a0_base = a_ptr.add(i * n + nc);
                let a1_base = a_ptr.add((i + 1) * n + nc);

                let mut j = jc;
                while j + 4 <= jc_end {
                    let b0_base = b_ptr.add((j + 0) * n + nc);
                    let b1_base = b_ptr.add((j + 1) * n + nc);
                    let b2_base = b_ptr.add((j + 2) * n + nc);
                    let b3_base = b_ptr.add((j + 3) * n + nc);

                    let mut acc00 = _mm256_setzero_ps();
                    let mut acc01 = _mm256_setzero_ps();
                    let mut acc02 = _mm256_setzero_ps();
                    let mut acc03 = _mm256_setzero_ps();
                    let mut acc10 = _mm256_setzero_ps();
                    let mut acc11 = _mm256_setzero_ps();
                    let mut acc12 = _mm256_setzero_ps();
                    let mut acc13 = _mm256_setzero_ps();

                    let mut col = 0usize;
                    while col + 8 <= seg_len {
                        let a0 = _mm256_loadu_ps(a0_base.add(col));
                        let a1 = _mm256_loadu_ps(a1_base.add(col));
                        let b0 = _mm256_loadu_ps(b0_base.add(col));
                        let b1 = _mm256_loadu_ps(b1_base.add(col));
                        let b2 = _mm256_loadu_ps(b2_base.add(col));
                        let b3 = _mm256_loadu_ps(b3_base.add(col));

                        acc00 = _mm256_fmadd_ps(a0, b0, acc00);
                        acc01 = _mm256_fmadd_ps(a0, b1, acc01);
                        acc02 = _mm256_fmadd_ps(a0, b2, acc02);
                        acc03 = _mm256_fmadd_ps(a0, b3, acc03);
                        acc10 = _mm256_fmadd_ps(a1, b0, acc10);
                        acc11 = _mm256_fmadd_ps(a1, b1, acc11);
                        acc12 = _mm256_fmadd_ps(a1, b2, acc12);
                        acc13 = _mm256_fmadd_ps(a1, b3, acc13);
                        col += 8;
                    }

                    let mut s00 = hsum256_ps(acc00);
                    let mut s01 = hsum256_ps(acc01);
                    let mut s02 = hsum256_ps(acc02);
                    let mut s03 = hsum256_ps(acc03);
                    let mut s10 = hsum256_ps(acc10);
                    let mut s11 = hsum256_ps(acc11);
                    let mut s12 = hsum256_ps(acc12);
                    let mut s13 = hsum256_ps(acc13);

                    while col < seg_len {
                        let av0 = *a0_base.add(col);
                        let av1 = *a1_base.add(col);
                        let bv0 = *b0_base.add(col);
                        let bv1 = *b1_base.add(col);
                        let bv2 = *b2_base.add(col);
                        let bv3 = *b3_base.add(col);
                        s00 += av0 * bv0;
                        s01 += av0 * bv1;
                        s02 += av0 * bv2;
                        s03 += av0 * bv3;
                        s10 += av1 * bv0;
                        s11 += av1 * bv1;
                        s12 += av1 * bv2;
                        s13 += av1 * bv3;
                        col += 1;
                    }

                    *c_ptr.add(i * k + j) += s00;
                    *c_ptr.add(i * k + j + 1) += s01;
                    *c_ptr.add(i * k + j + 2) += s02;
                    *c_ptr.add(i * k + j + 3) += s03;
                    *c_ptr.add((i + 1) * k + j) += s10;
                    *c_ptr.add((i + 1) * k + j + 1) += s11;
                    *c_ptr.add((i + 1) * k + j + 2) += s12;
                    *c_ptr.add((i + 1) * k + j + 3) += s13;
                    j += 4;
                }

                while j < jc_end {
                    let bj = b_ptr.add(j * n + nc);
                    *c_ptr.add(i * k + j) += dot_avx2_fma(a0_base, bj, seg_len);
                    *c_ptr.add((i + 1) * k + j) += dot_avx2_fma(a1_base, bj, seg_len);
                    j += 1;
                }
                i += 2;
            }

            if i < m {
                let ai = a_ptr.add(i * n + nc);
                for j in jc..jc_end {
                    let bj = b_ptr.add(j * n + nc);
                    *c_ptr.add(i * k + j) += dot_avx2_fma(ai, bj, seg_len);
                }
            }
        }
    }
}
