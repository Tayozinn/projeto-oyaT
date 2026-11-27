//! Gerador Determinístico de Bytecode SPIR-V 1.3 em Rust Puro (`SM-006` — Etapa 7).
//!
//! Constrói todos os shaders Compute necessários para o Transformer Decoder-only diretamente
//! em memória como fluxos de palavras `u32` conformes à especificação Khronos SPIR-V 1.3
//! (nativa em Vulkan 1.1+), eliminando qualquer dependência de `glslc`, `shaderc`, `CMake`
//! ou `VULKAN_SDK` na máquina do desenvolvedor ou do usuário final.
//!
//! ### ABI Uniforme de Todos os Shaders (`ServerMind Vulkan Compute ABI`)
//! - `DescriptorSet 0, Binding 0`: `StorageBuffer` `ArenaBuf { float data[]; }` (arena residente em VRAM)
//! - `DescriptorSet 0, Binding 1`: `StorageBuffer` `IoBuf { uint words[]; }` (IDs de entrada/alvo e métricas de saída)
//! - `PushConstant`: `PushParams { uint p[10]; }` (`40` bytes; parâmetros `f32` trafegam via `f32::to_bits()`
//!   e são convertidos sem custo via `OpBitcast`).

use std::cell::RefCell;
use std::collections::HashMap;

const SPV_MAGIC: u32 = 0x0723_0203;
const SPV_VERSION_1_3: u32 = 0x0001_0300;

// Opcodes SPIR-V oficiais (Khronos `spirv.h`)
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_BITCAST: u16 = 124;
const OP_IADD: u16 = 128;
const OP_FADD: u16 = 129;
const OP_ISUB: u16 = 130;
const OP_FSUB: u16 = 131;
const OP_IMUL: u16 = 132;
const OP_FMUL: u16 = 133;
const OP_UDIV: u16 = 134;
const OP_FDIV: u16 = 136;
const OP_UMOD: u16 = 137;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_SELECT: u16 = 169;
const OP_IEQUAL: u16 = 170;
const OP_ULESS_THAN: u16 = 176;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;

// Enumerações SPIR-V
const CAP_SHADER: u32 = 1;
const ADDR_LOGICAL: u32 = 0;
const MEM_GLSL450: u32 = 1;
const EXEC_MODEL_GLCOMPUTE: u32 = 5;
const EXEC_MODE_LOCAL_SIZE: u32 = 17;

const SC_INPUT: u32 = 1;
const SC_WORKGROUP: u32 = 4;
const SC_FUNCTION: u32 = 7;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_STORAGE_BUFFER: u32 = 12;

const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

const SCOPE_WORKGROUP: u32 = 2;
const SEM_WORKGROUP_ACQ_REL: u32 = 0x0000_0108; // AcquireRelease (0x8) | WorkgroupMemory (0x100)

// Instruções estendidas `GLSL.std.450`
#[allow(dead_code)]
const GLSL_FABS: u32 = 4;
const GLSL_TANH: u32 = 21;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;

pub const PUSH_CONSTANT_WORDS: usize = 10;
pub const PUSH_CONSTANT_BYTES: u32 = (PUSH_CONSTANT_WORDS * 4) as u32;

/// Coleção completa de módulos SPIR-V compilados para a sessão GPU.
pub struct NeuralSpirvKernels {
    pub fill_f32: Vec<u32>,
    pub vec_add_inplace: Vec<u32>,
    pub embed_fwd: Vec<u32>,
    pub embed_bwd: Vec<u32>,
    pub rmsnorm_fwd: Vec<u32>,
    pub rmsnorm_bwd_dx: Vec<u32>,
    pub rmsnorm_bwd_dgamma: Vec<u32>,
    pub matmul_nn_tiled: Vec<u32>,
    pub matmul_tn_tiled: Vec<u32>,
    pub matmul_nt_tiled: Vec<u32>,
    pub causal_mha_fwd: Vec<u32>,
    pub causal_mha_bwd_q_ds: Vec<u32>,
    pub causal_mha_bwd_kv: Vec<u32>,
    pub bias_gelu_fwd: Vec<u32>,
    pub gelu_bwd: Vec<u32>,
    pub bias_res_add_fwd: Vec<u32>,
    pub bias_grad_sum: Vec<u32>,
    pub cross_entropy_fwd_bwd: Vec<u32>,
    pub reduce_sum_f32_to_io: Vec<u32>,
    pub scale_and_grad_sq: Vec<u32>,
    pub adamw_step: Vec<u32>,
}

impl NeuralSpirvKernels {
    pub fn build_all() -> Self {
        Self {
            fill_f32: build_fill_f32(),
            vec_add_inplace: build_vec_add_inplace(),
            embed_fwd: build_embed_fwd(),
            embed_bwd: build_embed_bwd(),
            rmsnorm_fwd: build_rmsnorm_fwd(),
            rmsnorm_bwd_dx: build_rmsnorm_bwd_dx(),
            rmsnorm_bwd_dgamma: build_rmsnorm_bwd_dgamma(),
            matmul_nn_tiled: build_matmul_nn_tiled(),
            matmul_tn_tiled: build_matmul_tn_tiled(),
            matmul_nt_tiled: build_matmul_nt_tiled(),
            causal_mha_fwd: build_causal_mha_fwd(),
            causal_mha_bwd_q_ds: build_causal_mha_bwd_q_ds(),
            causal_mha_bwd_kv: build_causal_mha_bwd_kv(),
            bias_gelu_fwd: build_bias_gelu_fwd(),
            gelu_bwd: build_gelu_bwd(),
            bias_res_add_fwd: build_bias_res_add_fwd(),
            bias_grad_sum: build_bias_grad_sum(),
            cross_entropy_fwd_bwd: build_cross_entropy_fwd_bwd(),
            reduce_sum_f32_to_io: build_reduce_sum_f32_to_io(),
            scale_and_grad_sq: build_scale_and_grad_sq(),
            adamw_step: build_adamw_step(),
        }
    }
}

#[derive(Clone, Copy)]
struct SharedArrayDecl {
    len_const_id: u32,
    arr_type_id: u32,
    ptr_arr_type_id: u32,
    var_id: u32,
}

#[derive(Default)]
struct SpvMutable {
    next_id: u32,
    u32_consts: HashMap<u32, u32>,
    f32_consts: HashMap<u32, u32>,
    const_defs: Vec<(u16, Vec<u32>)>,
    shared_arrays: Vec<SharedArrayDecl>,
    local_vars: Vec<u32>,
    body: Vec<u32>,
}

struct SpvBuilder {
    local_size: (u32, u32, u32),
    ext_glsl: u32,
    fn_main: u32,
    entry_lbl: u32,

    // Tipos fundamentais
    ty_void: u32,
    ty_fn_void: u32,
    ty_bool: u32,
    ty_u32: u32,
    ty_f32: u32,
    ty_uvec3: u32,

    // Ponteiros
    ty_ptr_in_uvec3: u32,
    ty_ptr_in_u32: u32,
    ty_ptr_fn_u32: u32,
    ty_ptr_fn_f32: u32,
    ty_ptr_sb_f32: u32,
    ty_ptr_sb_u32: u32,
    ty_ptr_pc_u32: u32,
    ty_ptr_wg_f32: u32,

    // Estruturas e variáveis globais de interface
    ty_rt_f32: u32,
    ty_arena_struct: u32,
    ty_ptr_sb_arena: u32,
    var_arena: u32,

    ty_rt_u32: u32,
    ty_io_struct: u32,
    ty_ptr_sb_io: u32,
    var_io: u32,

    ty_arr_pc: u32,
    ty_pc_struct: u32,
    ty_ptr_pc_struct: u32,
    var_push: u32,

    var_gid: u32,
    var_lid: u32,
    var_wgid: u32,

    st: RefCell<SpvMutable>,
}

impl SpvBuilder {
    fn new(lx: u32, ly: u32, lz: u32) -> Self {
        let mut s = Self {
            local_size: (lx, ly, lz),
            ext_glsl: 0,
            fn_main: 0,
            entry_lbl: 0,
            ty_void: 0,
            ty_fn_void: 0,
            ty_bool: 0,
            ty_u32: 0,
            ty_f32: 0,
            ty_uvec3: 0,
            ty_ptr_in_uvec3: 0,
            ty_ptr_in_u32: 0,
            ty_ptr_fn_u32: 0,
            ty_ptr_fn_f32: 0,
            ty_ptr_sb_f32: 0,
            ty_ptr_sb_u32: 0,
            ty_ptr_pc_u32: 0,
            ty_ptr_wg_f32: 0,
            ty_rt_f32: 0,
            ty_arena_struct: 0,
            ty_ptr_sb_arena: 0,
            var_arena: 0,
            ty_rt_u32: 0,
            ty_io_struct: 0,
            ty_ptr_sb_io: 0,
            var_io: 0,
            ty_arr_pc: 0,
            ty_pc_struct: 0,
            ty_ptr_pc_struct: 0,
            var_push: 0,
            var_gid: 0,
            var_lid: 0,
            var_wgid: 0,
            st: RefCell::new(SpvMutable {
                next_id: 1,
                ..SpvMutable::default()
            }),
        };

        s.ext_glsl = s.alloc_id();
        s.fn_main = s.alloc_id();
        s.entry_lbl = s.alloc_id();

        s.ty_void = s.alloc_id();
        s.ty_fn_void = s.alloc_id();
        s.ty_bool = s.alloc_id();
        s.ty_u32 = s.alloc_id();
        s.ty_f32 = s.alloc_id();
        s.ty_uvec3 = s.alloc_id();

        s.ty_ptr_in_uvec3 = s.alloc_id();
        s.ty_ptr_in_u32 = s.alloc_id();
        s.ty_ptr_fn_u32 = s.alloc_id();
        s.ty_ptr_fn_f32 = s.alloc_id();
        s.ty_ptr_sb_f32 = s.alloc_id();
        s.ty_ptr_sb_u32 = s.alloc_id();
        s.ty_ptr_pc_u32 = s.alloc_id();
        s.ty_ptr_wg_f32 = s.alloc_id();

        s.ty_rt_f32 = s.alloc_id();
        s.ty_arena_struct = s.alloc_id();
        s.ty_ptr_sb_arena = s.alloc_id();
        s.var_arena = s.alloc_id();

        s.ty_rt_u32 = s.alloc_id();
        s.ty_io_struct = s.alloc_id();
        s.ty_ptr_sb_io = s.alloc_id();
        s.var_io = s.alloc_id();

        s.ty_arr_pc = s.alloc_id();
        s.ty_pc_struct = s.alloc_id();
        s.ty_ptr_pc_struct = s.alloc_id();
        s.var_push = s.alloc_id();

        s.var_gid = s.alloc_id();
        s.var_lid = s.alloc_id();
        s.var_wgid = s.alloc_id();

        // Pré-registra constantes estruturais obrigatórias
        let _ = s.const_u32(0);
        let _ = s.const_u32(PUSH_CONSTANT_WORDS as u32);
        s
    }

    #[inline]
    fn alloc_id(&self) -> u32 {
        let mut m = self.st.borrow_mut();
        let id = m.next_id;
        m.next_id += 1;
        id
    }

    fn const_u32(&self, val: u32) -> u32 {
        let mut m = self.st.borrow_mut();
        if let Some(&id) = m.u32_consts.get(&val) {
            return id;
        }
        let id = m.next_id;
        m.next_id += 1;
        m.u32_consts.insert(val, id);
        m.const_defs.push((OP_CONSTANT, vec![self.ty_u32, id, val]));
        id
    }

    fn const_f32(&self, val: f32) -> u32 {
        let bits = val.to_bits();
        let mut m = self.st.borrow_mut();
        if let Some(&id) = m.f32_consts.get(&bits) {
            return id;
        }
        let id = m.next_id;
        m.next_id += 1;
        m.f32_consts.insert(bits, id);
        m.const_defs
            .push((OP_CONSTANT, vec![self.ty_f32, id, bits]));
        id
    }

    fn decl_shared_f32_array(&self, len: u32) -> u32 {
        let len_const_id = self.const_u32(len);
        let arr_type_id = self.alloc_id();
        let ptr_arr_type_id = self.alloc_id();
        let var_id = self.alloc_id();
        self.st.borrow_mut().shared_arrays.push(SharedArrayDecl {
            len_const_id,
            arr_type_id,
            ptr_arr_type_id,
            var_id,
        });
        var_id
    }

    fn emit_inst(buf: &mut Vec<u32>, op: u16, args: &[u32]) {
        let wc = (1 + args.len()) as u32;
        buf.push((wc << 16) | (op as u32));
        buf.extend_from_slice(args);
    }

    fn emit_body(&self, op: u16, args: &[u32]) {
        Self::emit_inst(&mut self.st.borrow_mut().body, op, args);
    }

    // --- Acesso a Built-ins, PushConstants, Arena, IO e Shared Memory ---

    fn global_id(&self, comp: u32) -> u32 {
        let c = self.const_u32(comp);
        let ptr = self.alloc_id();
        self.emit_body(OP_ACCESS_CHAIN, &[self.ty_ptr_in_u32, ptr, self.var_gid, c]);
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, res, ptr]);
        res
    }

    fn local_id(&self, comp: u32) -> u32 {
        let c = self.const_u32(comp);
        let ptr = self.alloc_id();
        self.emit_body(OP_ACCESS_CHAIN, &[self.ty_ptr_in_u32, ptr, self.var_lid, c]);
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, res, ptr]);
        res
    }

    fn workgroup_id(&self, comp: u32) -> u32 {
        let c = self.const_u32(comp);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_in_u32, ptr, self.var_wgid, c],
        );
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, res, ptr]);
        res
    }

    fn push_u32(&self, idx: u32) -> u32 {
        assert!((idx as usize) < PUSH_CONSTANT_WORDS);
        let z = self.const_u32(0);
        let c = self.const_u32(idx);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_pc_u32, ptr, self.var_push, z, c],
        );
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, res, ptr]);
        res
    }

    fn push_f32(&self, idx: u32) -> u32 {
        let raw = self.push_u32(idx);
        let res = self.alloc_id();
        self.emit_body(OP_BITCAST, &[self.ty_f32, res, raw]);
        res
    }

    fn arena_load(&self, idx: u32) -> u32 {
        let z = self.const_u32(0);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_sb_f32, ptr, self.var_arena, z, idx],
        );
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_f32, res, ptr]);
        res
    }

    fn arena_store(&self, idx: u32, val: u32) {
        let z = self.const_u32(0);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_sb_f32, ptr, self.var_arena, z, idx],
        );
        self.emit_body(OP_STORE, &[ptr, val]);
    }

    fn arena_add_assign(&self, idx: u32, delta: u32) {
        let cur = self.arena_load(idx);
        let sum = self.fadd(cur, delta);
        self.arena_store(idx, sum);
    }

    fn io_load_u32(&self, idx: u32) -> u32 {
        let z = self.const_u32(0);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_sb_u32, ptr, self.var_io, z, idx],
        );
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, res, ptr]);
        res
    }

    fn io_store_f32_as_bits(&self, idx: u32, val_f32: u32) {
        let bits = self.alloc_id();
        self.emit_body(OP_BITCAST, &[self.ty_u32, bits, val_f32]);
        let z = self.const_u32(0);
        let ptr = self.alloc_id();
        self.emit_body(
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_sb_u32, ptr, self.var_io, z, idx],
        );
        self.emit_body(OP_STORE, &[ptr, bits]);
    }

    fn shared_load(&self, shared_var: u32, idx: u32) -> u32 {
        let ptr = self.alloc_id();
        self.emit_body(OP_ACCESS_CHAIN, &[self.ty_ptr_wg_f32, ptr, shared_var, idx]);
        let res = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_f32, res, ptr]);
        res
    }

    fn shared_store(&self, shared_var: u32, idx: u32, val: u32) {
        let ptr = self.alloc_id();
        self.emit_body(OP_ACCESS_CHAIN, &[self.ty_ptr_wg_f32, ptr, shared_var, idx]);
        self.emit_body(OP_STORE, &[ptr, val]);
    }

    fn workgroup_barrier(&self) {
        let scope = self.const_u32(SCOPE_WORKGROUP);
        let sem = self.const_u32(SEM_WORKGROUP_ACQ_REL);
        self.emit_body(OP_CONTROL_BARRIER, &[scope, scope, sem]);
    }

    // --- Variáveis locais na função `main` ---

    fn var_u32(&self, init: u32) -> u32 {
        let v = self.alloc_id();
        Self::emit_inst(
            &mut self.st.borrow_mut().local_vars,
            OP_VARIABLE,
            &[self.ty_ptr_fn_u32, v, SC_FUNCTION],
        );
        self.emit_body(OP_STORE, &[v, init]);
        v
    }

    fn var_f32(&self, init: u32) -> u32 {
        let v = self.alloc_id();
        Self::emit_inst(
            &mut self.st.borrow_mut().local_vars,
            OP_VARIABLE,
            &[self.ty_ptr_fn_f32, v, SC_FUNCTION],
        );
        self.emit_body(OP_STORE, &[v, init]);
        v
    }

    fn load_u32(&self, var: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_u32, r, var]);
        r
    }

    fn load_f32(&self, var: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_LOAD, &[self.ty_f32, r, var]);
        r
    }

    fn store(&self, var: u32, val: u32) {
        self.emit_body(OP_STORE, &[var, val]);
    }

    // --- Operações Aritméticas, Relacionais e Estendidas ---

    fn iadd(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_IADD, &[self.ty_u32, r, a, b]);
        r
    }

    fn isub(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_ISUB, &[self.ty_u32, r, a, b]);
        r
    }

    fn imul(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_IMUL, &[self.ty_u32, r, a, b]);
        r
    }

    fn udiv(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_UDIV, &[self.ty_u32, r, a, b]);
        r
    }

    fn umod(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_UMOD, &[self.ty_u32, r, a, b]);
        r
    }

    fn u_to_f32(&self, a: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_CONVERT_U_TO_F, &[self.ty_f32, r, a]);
        r
    }

    fn fadd(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_FADD, &[self.ty_f32, r, a, b]);
        r
    }

    fn fsub(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_FSUB, &[self.ty_f32, r, a, b]);
        r
    }

    fn fmul(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_FMUL, &[self.ty_f32, r, a, b]);
        r
    }

    fn fdiv(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_FDIV, &[self.ty_f32, r, a, b]);
        r
    }

    fn ult(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_ULESS_THAN, &[self.ty_bool, r, a, b]);
        r
    }

    fn ieq(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_IEQUAL, &[self.ty_bool, r, a, b]);
        r
    }

    fn fgt(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_FORD_GREATER_THAN, &[self.ty_bool, r, a, b]);
        r
    }

    fn logical_and(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_LOGICAL_AND, &[self.ty_bool, r, a, b]);
        r
    }

    fn logical_or(&self, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_LOGICAL_OR, &[self.ty_bool, r, a, b]);
        r
    }

    fn select_u32(&self, cond: u32, t_val: u32, f_val: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_SELECT, &[self.ty_u32, r, cond, t_val, f_val]);
        r
    }

    fn select_f32(&self, cond: u32, t_val: u32, f_val: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_SELECT, &[self.ty_f32, r, cond, t_val, f_val]);
        r
    }

    fn glsl_unary(&self, op_id: u32, x: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_EXT_INST, &[self.ty_f32, r, self.ext_glsl, op_id, x]);
        r
    }

    fn glsl_binary(&self, op_id: u32, x: u32, y: u32) -> u32 {
        let r = self.alloc_id();
        self.emit_body(OP_EXT_INST, &[self.ty_f32, r, self.ext_glsl, op_id, x, y]);
        r
    }

    #[allow(dead_code)]
    fn fabs(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_FABS, x)
    }

    fn ftanh(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_TANH, x)
    }

    fn fexp(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_EXP, x)
    }

    fn flog(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_LOG, x)
    }

    fn fsqrt(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_SQRT, x)
    }

    fn frsqrt(&self, x: u32) -> u32 {
        self.glsl_unary(GLSL_INVERSE_SQRT, x)
    }

    #[allow(dead_code)]
    fn fmin(&self, x: u32, y: u32) -> u32 {
        self.glsl_binary(GLSL_FMIN, x, y)
    }

    fn fmax(&self, x: u32, y: u32) -> u32 {
        self.glsl_binary(GLSL_FMAX, x, y)
    }

    // --- Controle de Fluxo Estruturado SPIR-V ---

    fn if_then<F>(&self, cond: u32, f: F)
    where
        F: FnOnce(&Self),
    {
        let then_lbl = self.alloc_id();
        let merge_lbl = self.alloc_id();
        self.emit_body(OP_SELECTION_MERGE, &[merge_lbl, 0]);
        self.emit_body(OP_BRANCH_CONDITIONAL, &[cond, then_lbl, merge_lbl]);
        self.emit_body(OP_LABEL, &[then_lbl]);
        f(self);
        self.emit_body(OP_BRANCH, &[merge_lbl]);
        self.emit_body(OP_LABEL, &[merge_lbl]);
    }

    fn if_else<Ft, Fe>(&self, cond: u32, f_true: Ft, f_false: Fe)
    where
        Ft: FnOnce(&Self),
        Fe: FnOnce(&Self),
    {
        let then_lbl = self.alloc_id();
        let else_lbl = self.alloc_id();
        let merge_lbl = self.alloc_id();
        self.emit_body(OP_SELECTION_MERGE, &[merge_lbl, 0]);
        self.emit_body(OP_BRANCH_CONDITIONAL, &[cond, then_lbl, else_lbl]);

        self.emit_body(OP_LABEL, &[then_lbl]);
        f_true(self);
        self.emit_body(OP_BRANCH, &[merge_lbl]);

        self.emit_body(OP_LABEL, &[else_lbl]);
        f_false(self);
        self.emit_body(OP_BRANCH, &[merge_lbl]);

        self.emit_body(OP_LABEL, &[merge_lbl]);
    }

    fn for_range<F>(&self, start: u32, end: u32, step: u32, mut f: F)
    where
        F: FnMut(&Self, u32),
    {
        let iv = self.var_u32(start);
        let hdr_lbl = self.alloc_id();
        let body_lbl = self.alloc_id();
        let cont_lbl = self.alloc_id();
        let merge_lbl = self.alloc_id();

        self.emit_body(OP_BRANCH, &[hdr_lbl]);
        self.emit_body(OP_LABEL, &[hdr_lbl]);
        let cur = self.load_u32(iv);
        let cond = self.ult(cur, end);
        self.emit_body(OP_LOOP_MERGE, &[merge_lbl, cont_lbl, 0]);
        self.emit_body(OP_BRANCH_CONDITIONAL, &[cond, body_lbl, merge_lbl]);

        self.emit_body(OP_LABEL, &[body_lbl]);
        f(self, cur);
        self.emit_body(OP_BRANCH, &[cont_lbl]);

        self.emit_body(OP_LABEL, &[cont_lbl]);
        let cur2 = self.load_u32(iv);
        let next = self.iadd(cur2, step);
        self.store(iv, next);
        self.emit_body(OP_BRANCH, &[hdr_lbl]);

        self.emit_body(OP_LABEL, &[merge_lbl]);
    }

    /// Monta o módulo SPIR-V completo obedecendo estritamente à ordem lógica da seção 2.4 da especificação.
    fn finalize(self) -> Vec<u32> {
        let m = self.st.into_inner();
        let mut out = Vec::with_capacity(256 + m.local_vars.len() + m.body.len());

        // 1. Cabeçalho (5 palavras)
        out.push(SPV_MAGIC);
        out.push(SPV_VERSION_1_3);
        out.push(0); // Generator
        out.push(m.next_id); // Bound
        out.push(0); // Schema

        // 2. Capabilities
        Self::emit_inst(&mut out, OP_CAPABILITY, &[CAP_SHADER]);

        // 3. ExtInstImport ("GLSL.std.450")
        let mut ext_args = vec![self.ext_glsl];
        ext_args.extend(encode_spv_string("GLSL.std.450"));
        Self::emit_inst(&mut out, OP_EXT_INST_IMPORT, &ext_args);

        // 4. MemoryModel
        Self::emit_inst(&mut out, OP_MEMORY_MODEL, &[ADDR_LOGICAL, MEM_GLSL450]);

        // 5. EntryPoint (SPIR-V 1.3: lista exclusivamente variáveis de interface Input/Output)
        let mut ep_args = vec![EXEC_MODEL_GLCOMPUTE, self.fn_main];
        ep_args.extend(encode_spv_string("main"));
        ep_args.extend_from_slice(&[self.var_gid, self.var_lid, self.var_wgid]);
        Self::emit_inst(&mut out, OP_ENTRY_POINT, &ep_args);

        // 6. ExecutionMode (LocalSize)
        Self::emit_inst(
            &mut out,
            OP_EXECUTION_MODE,
            &[
                self.fn_main,
                EXEC_MODE_LOCAL_SIZE,
                self.local_size.0,
                self.local_size.1,
                self.local_size.2,
            ],
        );

        // 7. Anotações / Decorations
        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.var_gid, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID],
        );
        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.var_lid, DEC_BUILTIN, BUILTIN_LOCAL_INVOCATION_ID],
        );
        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.var_wgid, DEC_BUILTIN, BUILTIN_WORKGROUP_ID],
        );

        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.ty_rt_f32, DEC_ARRAY_STRIDE, 4],
        );
        Self::emit_inst(&mut out, OP_DECORATE, &[self.ty_arena_struct, DEC_BLOCK]);
        Self::emit_inst(
            &mut out,
            OP_MEMBER_DECORATE,
            &[self.ty_arena_struct, 0, DEC_OFFSET, 0],
        );
        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.var_arena, DEC_DESCRIPTOR_SET, 0],
        );
        Self::emit_inst(&mut out, OP_DECORATE, &[self.var_arena, DEC_BINDING, 0]);

        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.ty_rt_u32, DEC_ARRAY_STRIDE, 4],
        );
        Self::emit_inst(&mut out, OP_DECORATE, &[self.ty_io_struct, DEC_BLOCK]);
        Self::emit_inst(
            &mut out,
            OP_MEMBER_DECORATE,
            &[self.ty_io_struct, 0, DEC_OFFSET, 0],
        );
        Self::emit_inst(&mut out, OP_DECORATE, &[self.var_io, DEC_DESCRIPTOR_SET, 0]);
        Self::emit_inst(&mut out, OP_DECORATE, &[self.var_io, DEC_BINDING, 1]);

        Self::emit_inst(
            &mut out,
            OP_DECORATE,
            &[self.ty_arr_pc, DEC_ARRAY_STRIDE, 4],
        );
        Self::emit_inst(&mut out, OP_DECORATE, &[self.ty_pc_struct, DEC_BLOCK]);
        Self::emit_inst(
            &mut out,
            OP_MEMBER_DECORATE,
            &[self.ty_pc_struct, 0, DEC_OFFSET, 0],
        );

        // 8. Tipos, Constantes e Variáveis Globais (em ordem de dependência estrita)
        Self::emit_inst(&mut out, OP_TYPE_VOID, &[self.ty_void]);
        Self::emit_inst(&mut out, OP_TYPE_FUNCTION, &[self.ty_fn_void, self.ty_void]);
        Self::emit_inst(&mut out, OP_TYPE_BOOL, &[self.ty_bool]);
        Self::emit_inst(&mut out, OP_TYPE_INT, &[self.ty_u32, 32, 0]);
        Self::emit_inst(&mut out, OP_TYPE_FLOAT, &[self.ty_f32, 32]);
        Self::emit_inst(&mut out, OP_TYPE_VECTOR, &[self.ty_uvec3, self.ty_u32, 3]);

        // Constantes escalares (`u32` e `f32`)
        for (op, args) in &m.const_defs {
            Self::emit_inst(&mut out, *op, args);
        }

        // Ponteiros básicos
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_in_uvec3, SC_INPUT, self.ty_uvec3],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_in_u32, SC_INPUT, self.ty_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_fn_u32, SC_FUNCTION, self.ty_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_fn_f32, SC_FUNCTION, self.ty_f32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_sb_f32, SC_STORAGE_BUFFER, self.ty_f32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_sb_u32, SC_STORAGE_BUFFER, self.ty_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_pc_u32, SC_PUSH_CONSTANT, self.ty_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_wg_f32, SC_WORKGROUP, self.ty_f32],
        );

        // Variáveis BuiltIn Input
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_in_uvec3, self.var_gid, SC_INPUT],
        );
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_in_uvec3, self.var_lid, SC_INPUT],
        );
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_in_uvec3, self.var_wgid, SC_INPUT],
        );

        // StorageBuffer 0: Arena f32[]
        Self::emit_inst(
            &mut out,
            OP_TYPE_RUNTIME_ARRAY,
            &[self.ty_rt_f32, self.ty_f32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_STRUCT,
            &[self.ty_arena_struct, self.ty_rt_f32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[
                self.ty_ptr_sb_arena,
                SC_STORAGE_BUFFER,
                self.ty_arena_struct,
            ],
        );
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_sb_arena, self.var_arena, SC_STORAGE_BUFFER],
        );

        // StorageBuffer 1: IO u32[]
        Self::emit_inst(
            &mut out,
            OP_TYPE_RUNTIME_ARRAY,
            &[self.ty_rt_u32, self.ty_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_STRUCT,
            &[self.ty_io_struct, self.ty_rt_u32],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_sb_io, SC_STORAGE_BUFFER, self.ty_io_struct],
        );
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_sb_io, self.var_io, SC_STORAGE_BUFFER],
        );

        // PushConstants: u32[10]
        let pc_len_const = *m
            .u32_consts
            .get(&(PUSH_CONSTANT_WORDS as u32))
            .expect("pc_len_const");
        Self::emit_inst(
            &mut out,
            OP_TYPE_ARRAY,
            &[self.ty_arr_pc, self.ty_u32, pc_len_const],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_STRUCT,
            &[self.ty_pc_struct, self.ty_arr_pc],
        );
        Self::emit_inst(
            &mut out,
            OP_TYPE_POINTER,
            &[self.ty_ptr_pc_struct, SC_PUSH_CONSTANT, self.ty_pc_struct],
        );
        Self::emit_inst(
            &mut out,
            OP_VARIABLE,
            &[self.ty_ptr_pc_struct, self.var_push, SC_PUSH_CONSTANT],
        );

        // Arrays em memória compartilhada `Workgroup` (LDS)
        for sh in &m.shared_arrays {
            Self::emit_inst(
                &mut out,
                OP_TYPE_ARRAY,
                &[sh.arr_type_id, self.ty_f32, sh.len_const_id],
            );
            Self::emit_inst(
                &mut out,
                OP_TYPE_POINTER,
                &[sh.ptr_arr_type_id, SC_WORKGROUP, sh.arr_type_id],
            );
            Self::emit_inst(
                &mut out,
                OP_VARIABLE,
                &[sh.ptr_arr_type_id, sh.var_id, SC_WORKGROUP],
            );
        }

        // 9. Definição da Função `main`
        Self::emit_inst(
            &mut out,
            OP_FUNCTION,
            &[self.ty_void, self.fn_main, 0, self.ty_fn_void],
        );
        Self::emit_inst(&mut out, OP_LABEL, &[self.entry_lbl]);
        out.extend_from_slice(&m.local_vars);
        out.extend_from_slice(&m.body);
        Self::emit_inst(&mut out, OP_RETURN, &[]);
        Self::emit_inst(&mut out, OP_FUNCTION_END, &[]);

        out
    }
}

fn encode_spv_string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Redução paralela por soma em árvore dentro de um Workgroup de tamanho `wg_size` (potência de 2).
fn emit_wg_tree_sum(b: &SpvBuilder, shared: u32, lid: u32, wg_size: u32) {
    b.workgroup_barrier();
    let mut stride = wg_size / 2;
    while stride >= 1 {
        let c_stride = b.const_u32(stride);
        let active = b.ult(lid, c_stride);
        b.if_then(active, |b| {
            let mate = b.iadd(lid, c_stride);
            let a = b.shared_load(shared, lid);
            let m = b.shared_load(shared, mate);
            let s = b.fadd(a, m);
            b.shared_store(shared, lid, s);
        });
        b.workgroup_barrier();
        stride /= 2;
    }
}

/// Redução paralela por máximo (`FMax`) dentro de um Workgroup de tamanho `wg_size` (potência de 2).
fn emit_wg_tree_max(b: &SpvBuilder, shared: u32, lid: u32, wg_size: u32) {
    b.workgroup_barrier();
    let mut stride = wg_size / 2;
    while stride >= 1 {
        let c_stride = b.const_u32(stride);
        let active = b.ult(lid, c_stride);
        b.if_then(active, |b| {
            let mate = b.iadd(lid, c_stride);
            let a = b.shared_load(shared, lid);
            let m = b.shared_load(shared, mate);
            let mx = b.fmax(a, m);
            b.shared_store(shared, lid, mx);
        });
        b.workgroup_barrier();
        stride /= 2;
    }
}

/// Emite a função GELU aproximada de Hendrycks & Gimpel com saturação segura em `|x| >= 10.0` (`SM-001`).
fn emit_gelu_f32(b: &SpvBuilder, x: u32) -> u32 {
    let c_pos10 = b.const_f32(10.0);
    let c_neg10 = b.const_f32(-10.0);
    let c_zero = b.const_f32(0.0);
    let c_half = b.const_f32(0.5);
    let c_one = b.const_f32(1.0);
    let c_k = b.const_f32(0.7978845608);
    let c_c = b.const_f32(0.044715);

    // Clamp da entrada do polinômio cúbico em [-10, 10] evita overflow em `x^3`
    let x_hi = b.fmin(x, c_pos10);
    let xc = b.fmax(x_hi, c_neg10);
    let x2 = b.fmul(xc, xc);
    let x3 = b.fmul(x2, xc);
    let cx3 = b.fmul(c_c, x3);
    let sum = b.fadd(xc, cx3);
    let inner = b.fmul(c_k, sum);
    let th = b.ftanh(inner);
    let one_plus = b.fadd(c_one, th);
    let half_x = b.fmul(c_half, x);
    let approx = b.fmul(half_x, one_plus);

    let gt_pos10 = b.fgt(x, c_pos10);
    let lt_neg10 = b.fgt(c_neg10, x);
    let s1 = b.select_f32(gt_pos10, x, approx);
    b.select_f32(lt_neg10, c_zero, s1)
}

/// Emite a derivada analítica exata de `gelu(x)` com saturação segura em `|x| >= 10.0` (`SM-001`).
fn emit_gelu_grad_f32(b: &SpvBuilder, x: u32) -> u32 {
    let c_pos10 = b.const_f32(10.0);
    let c_neg10 = b.const_f32(-10.0);
    let c_zero = b.const_f32(0.0);
    let c_half = b.const_f32(0.5);
    let c_one = b.const_f32(1.0);
    let c_three = b.const_f32(3.0);
    let c_k = b.const_f32(0.7978845608);
    let c_c = b.const_f32(0.044715);

    let x_hi = b.fmin(x, c_pos10);
    let xc = b.fmax(x_hi, c_neg10);
    let x2 = b.fmul(xc, xc);
    let x3 = b.fmul(x2, xc);
    let cx3 = b.fmul(c_c, x3);
    let inner = b.fmul(c_k, b.fadd(xc, cx3));
    let t = b.ftanh(inner);
    let t2 = b.fmul(t, t);
    let sech2 = b.fsub(c_one, t2);
    let three_c = b.fmul(c_three, c_c);
    let term_x2 = b.fmul(three_c, x2);
    let d_inner = b.fmul(c_k, b.fadd(c_one, term_x2));

    let left = b.fmul(c_half, b.fadd(c_one, t));
    let right = b.fmul(b.fmul(c_half, xc), b.fmul(sech2, d_inner));
    let approx = b.fadd(left, right);

    let gt_pos10 = b.fgt(x, c_pos10);
    let lt_neg10 = b.fgt(c_neg10, x);
    let s1 = b.select_f32(gt_pos10, c_one, approx);
    b.select_f32(lt_neg10, c_zero, s1)
}

// ===========================================================================
// 1..4. Utilitários Básicos e Embeddings
// ===========================================================================

/// `p0 = off_dst, p1 = len, p2 = val_bits`
fn build_fill_f32() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_dst = b.push_u32(0);
    let len = b.push_u32(1);
    let val = b.push_f32(2);
    let cond = b.ult(gid, len);
    b.if_then(cond, |b| {
        let dst = b.iadd(off_dst, gid);
        b.arena_store(dst, val);
    });
    b.finalize()
}

/// `p0 = off_dst, p1 = off_src, p2 = len` -> `arena[off_dst + i] += arena[off_src + i]`
fn build_vec_add_inplace() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_dst = b.push_u32(0);
    let off_src = b.push_u32(1);
    let len = b.push_u32(2);
    let cond = b.ult(gid, len);
    b.if_then(cond, |b| {
        let di = b.iadd(off_dst, gid);
        let si = b.iadd(off_src, gid);
        let sv = b.arena_load(si);
        b.arena_add_assign(di, sv);
    });
    b.finalize()
}

/// `p0 = off_tok, p1 = off_pos, p2 = off_x, p3 = t, p4 = d, p5 = vocab, p6 = io_in_off`
fn build_embed_fwd() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_tok = b.push_u32(0);
    let off_pos = b.push_u32(1);
    let off_x = b.push_u32(2);
    let t = b.push_u32(3);
    let d = b.push_u32(4);
    let vocab = b.push_u32(5);
    let io_in = b.push_u32(6);
    let td = b.imul(t, d);

    let cond = b.ult(gid, td);
    b.if_then(cond, |b| {
        let pos = b.udiv(gid, d);
        let col = b.umod(gid, d);
        let io_idx = b.iadd(io_in, pos);
        let raw_tid = b.io_load_u32(io_idx);
        let one = b.const_u32(1);
        let vmax = b.isub(vocab, one);
        let in_range = b.ult(raw_tid, vocab);
        let tid = b.select_u32(in_range, raw_tid, vmax);

        let tok_idx = b.iadd(off_tok, b.iadd(b.imul(tid, d), col));
        let pos_idx = b.iadd(off_pos, gid);
        let x_idx = b.iadd(off_x, gid);
        let tv = b.arena_load(tok_idx);
        let pv = b.arena_load(pos_idx);
        let sum = b.fadd(tv, pv);
        b.arena_store(x_idx, sum);
    });
    b.finalize()
}

/// `p0 = off_dx, p1 = off_gtok, p2 = off_gpos, p3 = t, p4 = d, p5 = vocab, p6 = io_in_off`
/// Paraleliza por coluna `col = gid.x < d` iterando `pos in 0..t` (100% livre de corridas sem atomics).
fn build_embed_bwd() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let col = b.global_id(0);
    let off_dx = b.push_u32(0);
    let off_gtok = b.push_u32(1);
    let off_gpos = b.push_u32(2);
    let t = b.push_u32(3);
    let d = b.push_u32(4);
    let vocab = b.push_u32(5);
    let io_in = b.push_u32(6);

    let cond = b.ult(col, d);
    b.if_then(cond, |b| {
        let u0 = b.const_u32(0);
        let u1 = b.const_u32(1);
        let vmax = b.isub(vocab, u1);
        b.for_range(u0, t, u1, |b, pos| {
            let io_idx = b.iadd(io_in, pos);
            let raw_tid = b.io_load_u32(io_idx);
            let ok = b.ult(raw_tid, vocab);
            let tid = b.select_u32(ok, raw_tid, vmax);

            let row_col = b.iadd(b.imul(pos, d), col);
            let dx = b.arena_load(b.iadd(off_dx, row_col));
            let gtok_idx = b.iadd(off_gtok, b.iadd(b.imul(tid, d), col));
            let gpos_idx = b.iadd(off_gpos, row_col);
            b.arena_add_assign(gtok_idx, dx);
            b.arena_add_assign(gpos_idx, dx);
        });
    });
    b.finalize()
}

// ===========================================================================
// 5..6. RMSNorm Forward e Backward
// ===========================================================================

/// 1 Workgroup (`64` threads) por posição `pos = wgid.x < t`.
/// `p0 = off_x, p1 = off_gamma, p2 = off_out, p3 = off_inv_rms, p4 = t, p5 = d`
fn build_rmsnorm_fwd() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let shared = b.decl_shared_f32_array(64);
    let pos = b.workgroup_id(0);
    let lid = b.local_id(0);
    let off_x = b.push_u32(0);
    let off_gamma = b.push_u32(1);
    let off_out = b.push_u32(2);
    let off_inv = b.push_u32(3);
    let _t = b.push_u32(4);
    let d = b.push_u32(5);

    let f0 = b.const_f32(0.0);
    let eps = b.const_f32(1e-5);
    let c64 = b.const_u32(64);
    let u0 = b.const_u32(0);

    let row_off = b.imul(pos, d);
    let acc = b.var_f32(f0);
    b.for_range(lid, d, c64, |b, j| {
        let xv = b.arena_load(b.iadd(off_x, b.iadd(row_off, j)));
        let sq = b.fmul(xv, xv);
        let cur = b.load_f32(acc);
        b.store(acc, b.fadd(cur, sq));
    });
    let part = b.load_f32(acc);
    b.shared_store(shared, lid, part);
    emit_wg_tree_sum(&b, shared, lid, 64);

    let ss = b.shared_load(shared, u0);
    let df = b.u_to_f32(d);
    let mean_sq = b.fadd(b.fdiv(ss, df), eps);
    let ir = b.frsqrt(mean_sq);

    let is_l0 = b.ieq(lid, u0);
    b.if_then(is_l0, |b| {
        b.arena_store(b.iadd(off_inv, pos), ir);
    });

    b.for_range(lid, d, c64, |b, j| {
        let xv = b.arena_load(b.iadd(off_x, b.iadd(row_off, j)));
        let gv = b.arena_load(b.iadd(off_gamma, j));
        let yv = b.fmul(b.fmul(xv, ir), gv);
        b.arena_store(b.iadd(off_out, b.iadd(row_off, j)), yv);
    });

    b.finalize()
}

/// 1 Workgroup (`64` threads) por posição `pos = wgid.x < t` calculando `d_x`.
/// `p0 = off_x, p1 = off_gamma, p2 = off_inv, p3 = off_dout, p4 = off_dx, p5 = d`
fn build_rmsnorm_bwd_dx() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let shared = b.decl_shared_f32_array(64);
    let pos = b.workgroup_id(0);
    let lid = b.local_id(0);
    let off_x = b.push_u32(0);
    let off_gamma = b.push_u32(1);
    let off_inv = b.push_u32(2);
    let off_dout = b.push_u32(3);
    let off_dx = b.push_u32(4);
    let d = b.push_u32(5);

    let f0 = b.const_f32(0.0);
    let c64 = b.const_u32(64);
    let u0 = b.const_u32(0);
    let row_off = b.imul(pos, d);

    let acc = b.var_f32(f0);
    b.for_range(lid, d, c64, |b, j| {
        let xv = b.arena_load(b.iadd(off_x, b.iadd(row_off, j)));
        let gv = b.arena_load(b.iadd(off_gamma, j));
        let dov = b.arena_load(b.iadd(off_dout, b.iadd(row_off, j)));
        let prod = b.fmul(b.fmul(dov, gv), xv);
        let cur = b.load_f32(acc);
        b.store(acc, b.fadd(cur, prod));
    });
    let part = b.load_f32(acc);
    b.shared_store(shared, lid, part);
    emit_wg_tree_sum(&b, shared, lid, 64);

    let dot = b.shared_load(shared, u0);
    let ir = b.arena_load(b.iadd(off_inv, pos));
    let ir2 = b.fmul(ir, ir);
    let ir3 = b.fmul(ir2, ir);
    let df = b.u_to_f32(d);
    let coeff = b.fdiv(b.fmul(dot, ir3), df);

    b.for_range(lid, d, c64, |b, j| {
        let xv = b.arena_load(b.iadd(off_x, b.iadd(row_off, j)));
        let gv = b.arena_load(b.iadd(off_gamma, j));
        let dov = b.arena_load(b.iadd(off_dout, b.iadd(row_off, j)));
        let term1 = b.fmul(b.fmul(dov, gv), ir);
        let term2 = b.fmul(xv, coeff);
        let dxv = b.fsub(term1, term2);
        b.arena_store(b.iadd(off_dx, b.iadd(row_off, j)), dxv);
    });

    b.finalize()
}

/// 1 thread por coluna `col = gid.x < d` acumulando `d_gamma[col]`.
/// `p0 = off_x, p1 = off_inv, p2 = off_dout, p3 = off_dgamma, p4 = t, p5 = d`
fn build_rmsnorm_bwd_dgamma() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let col = b.global_id(0);
    let off_x = b.push_u32(0);
    let off_inv = b.push_u32(1);
    let off_dout = b.push_u32(2);
    let off_dgamma = b.push_u32(3);
    let t = b.push_u32(4);
    let d = b.push_u32(5);

    let cond = b.ult(col, d);
    b.if_then(cond, |b| {
        let f0 = b.const_f32(0.0);
        let u0 = b.const_u32(0);
        let u1 = b.const_u32(1);
        let sum = b.var_f32(f0);
        b.for_range(u0, t, u1, |b, pos| {
            let ir = b.arena_load(b.iadd(off_inv, pos));
            let rc = b.iadd(b.imul(pos, d), col);
            let xv = b.arena_load(b.iadd(off_x, rc));
            let dov = b.arena_load(b.iadd(off_dout, rc));
            let prod = b.fmul(b.fmul(dov, xv), ir);
            let cur = b.load_f32(sum);
            b.store(sum, b.fadd(cur, prod));
        });
        let total = b.load_f32(sum);
        b.arena_add_assign(b.iadd(off_dgamma, col), total);
    });
    b.finalize()
}

// ===========================================================================
// 7..9. Kernels GEMM Tiled 16×16 em LDS (`Workgroup` Shared Memory)
// ===========================================================================

/// `C[m, n] += A[m, k] @ B[k, n]`. Workgroup `(16, 16, 1)`.
/// `p0 = off_a, p1 = off_b, p2 = off_c, p3 = m, p4 = k, p5 = n`
fn build_matmul_nn_tiled() -> Vec<u32> {
    let b = SpvBuilder::new(16, 16, 1);
    let tile_a = b.decl_shared_f32_array(256);
    let tile_b = b.decl_shared_f32_array(256);

    let lx = b.local_id(0);
    let ly = b.local_id(1);
    let wx = b.workgroup_id(0);
    let wy = b.workgroup_id(1);

    let off_a = b.push_u32(0);
    let off_b = b.push_u32(1);
    let off_c = b.push_u32(2);
    let m = b.push_u32(3);
    let k = b.push_u32(4);
    let n = b.push_u32(5);

    let u0 = b.const_u32(0);
    let u1 = b.const_u32(1);
    let u15 = b.const_u32(15);
    let u16 = b.const_u32(16);
    let f0 = b.const_f32(0.0);

    let col = b.iadd(b.imul(wx, u16), lx);
    let row = b.iadd(b.imul(wy, u16), ly);
    let l_flat = b.iadd(b.imul(ly, u16), lx);
    let num_tiles = b.udiv(b.iadd(k, u15), u16);

    let acc = b.var_f32(f0);

    b.for_range(u0, num_tiles, u1, |b, t_idx| {
        let t_base = b.imul(t_idx, u16);
        let a_col = b.iadd(t_base, lx);
        let b_row = b.iadd(t_base, ly);

        let a_ok = b.logical_and(b.ult(row, m), b.ult(a_col, k));
        let av = b.var_f32(f0);
        b.if_then(a_ok, |b| {
            let idx = b.iadd(off_a, b.iadd(b.imul(row, k), a_col));
            let v = b.arena_load(idx);
            b.store(av, v);
        });
        let av_val = b.load_f32(av);
        b.shared_store(tile_a, l_flat, av_val);

        let b_ok = b.logical_and(b.ult(b_row, k), b.ult(col, n));
        let bv = b.var_f32(f0);
        b.if_then(b_ok, |b| {
            let idx = b.iadd(off_b, b.iadd(b.imul(b_row, n), col));
            let v = b.arena_load(idx);
            b.store(bv, v);
        });
        let bv_val = b.load_f32(bv);
        b.shared_store(tile_b, l_flat, bv_val);

        b.workgroup_barrier();

        b.for_range(u0, u16, u1, |b, p| {
            let ta = b.shared_load(tile_a, b.iadd(b.imul(ly, u16), p));
            let tb = b.shared_load(tile_b, b.iadd(b.imul(p, u16), lx));
            let cur = b.load_f32(acc);
            b.store(acc, b.fadd(cur, b.fmul(ta, tb)));
        });

        b.workgroup_barrier();
    });

    let c_ok = b.logical_and(b.ult(row, m), b.ult(col, n));
    b.if_then(c_ok, |b| {
        let total = b.load_f32(acc);
        let c_idx = b.iadd(off_c, b.iadd(b.imul(row, n), col));
        b.arena_add_assign(c_idx, total);
    });

    b.finalize()
}

/// `C[k, n] += A[m, k]^T @ B[m, n]`. Workgroup `(16, 16, 1)`.
/// `p0 = off_a, p1 = off_b, p2 = off_c, p3 = k, p4 = m, p5 = n`
fn build_matmul_tn_tiled() -> Vec<u32> {
    let b = SpvBuilder::new(16, 16, 1);
    let tile_a = b.decl_shared_f32_array(256);
    let tile_b = b.decl_shared_f32_array(256);

    let lx = b.local_id(0);
    let ly = b.local_id(1);
    let wx = b.workgroup_id(0);
    let wy = b.workgroup_id(1);

    let off_a = b.push_u32(0);
    let off_b = b.push_u32(1);
    let off_c = b.push_u32(2);
    let k = b.push_u32(3);
    let m = b.push_u32(4);
    let n = b.push_u32(5);

    let u0 = b.const_u32(0);
    let u1 = b.const_u32(1);
    let u15 = b.const_u32(15);
    let u16 = b.const_u32(16);
    let f0 = b.const_f32(0.0);

    // `row` indexa `0..k`, `col` indexa `0..n`
    let col = b.iadd(b.imul(wx, u16), lx);
    let row = b.iadd(b.imul(wy, u16), ly);
    let l_flat = b.iadd(b.imul(ly, u16), lx);
    let num_tiles = b.udiv(b.iadd(m, u15), u16);

    let acc = b.var_f32(f0);

    b.for_range(u0, num_tiles, u1, |b, t_idx| {
        let t_base = b.imul(t_idx, u16);
        // Carregamento coalescido de A[m_row, k_col] transposto em LDS:
        // thread (lx, ly) lê A[(t_base + ly), (wy*16 + lx)] e guarda em tile_a[lx * 16 + ly] (= A^T[wy*16+lx, t_base+ly])
        let a_m_row = b.iadd(t_base, ly);
        let a_k_col = b.iadd(b.imul(wy, u16), lx);
        let a_ok = b.logical_and(b.ult(a_m_row, m), b.ult(a_k_col, k));
        let av = b.var_f32(f0);
        b.if_then(a_ok, |b| {
            let idx = b.iadd(off_a, b.iadd(b.imul(a_m_row, k), a_k_col));
            let v = b.arena_load(idx);
            b.store(av, v);
        });
        let av_val = b.load_f32(av);
        let tr_flat = b.iadd(b.imul(lx, u16), ly);
        b.shared_store(tile_a, tr_flat, av_val);

        let b_m_row = b.iadd(t_base, ly);
        let b_ok = b.logical_and(b.ult(b_m_row, m), b.ult(col, n));
        let bv = b.var_f32(f0);
        b.if_then(b_ok, |b| {
            let idx = b.iadd(off_b, b.iadd(b.imul(b_m_row, n), col));
            let v = b.arena_load(idx);
            b.store(bv, v);
        });
        let bv_val = b.load_f32(bv);
        b.shared_store(tile_b, l_flat, bv_val);

        b.workgroup_barrier();

        b.for_range(u0, u16, u1, |b, p| {
            let ta = b.shared_load(tile_a, b.iadd(b.imul(ly, u16), p));
            let tb = b.shared_load(tile_b, b.iadd(b.imul(p, u16), lx));
            let cur = b.load_f32(acc);
            b.store(acc, b.fadd(cur, b.fmul(ta, tb)));
        });

        b.workgroup_barrier();
    });

    let c_ok = b.logical_and(b.ult(row, k), b.ult(col, n));
    b.if_then(c_ok, |b| {
        let total = b.load_f32(acc);
        let c_idx = b.iadd(off_c, b.iadd(b.imul(row, n), col));
        b.arena_add_assign(c_idx, total);
    });

    b.finalize()
}

/// `C[m, k] += A[m, n] @ B[k, n]^T`. Workgroup `(16, 16, 1)`.
/// `p0 = off_a, p1 = off_b, p2 = off_c, p3 = m, p4 = n, p5 = k`
fn build_matmul_nt_tiled() -> Vec<u32> {
    let b = SpvBuilder::new(16, 16, 1);
    let tile_a = b.decl_shared_f32_array(256);
    let tile_b = b.decl_shared_f32_array(256);

    let lx = b.local_id(0);
    let ly = b.local_id(1);
    let wx = b.workgroup_id(0);
    let wy = b.workgroup_id(1);

    let off_a = b.push_u32(0);
    let off_b = b.push_u32(1);
    let off_c = b.push_u32(2);
    let m = b.push_u32(3);
    let n = b.push_u32(4);
    let k = b.push_u32(5);

    let u0 = b.const_u32(0);
    let u1 = b.const_u32(1);
    let u15 = b.const_u32(15);
    let u16 = b.const_u32(16);
    let f0 = b.const_f32(0.0);

    // `row` em `0..m`, `col` em `0..k`
    let col = b.iadd(b.imul(wx, u16), lx);
    let row = b.iadd(b.imul(wy, u16), ly);
    let l_flat = b.iadd(b.imul(ly, u16), lx);
    let num_tiles = b.udiv(b.iadd(n, u15), u16);

    let acc = b.var_f32(f0);

    b.for_range(u0, num_tiles, u1, |b, t_idx| {
        let t_base = b.imul(t_idx, u16);
        let n_col = b.iadd(t_base, lx);

        let a_ok = b.logical_and(b.ult(row, m), b.ult(n_col, n));
        let av = b.var_f32(f0);
        b.if_then(a_ok, |b| {
            let idx = b.iadd(off_a, b.iadd(b.imul(row, n), n_col));
            let v = b.arena_load(idx);
            b.store(av, v);
        });
        let av_val = b.load_f32(av);
        b.shared_store(tile_a, l_flat, av_val);

        // Coalescido em B[k, n]: thread (lx, ly) lê B[wx*16 + ly, t_base + lx]
        let b_k_row = b.iadd(b.imul(wx, u16), ly);
        let b_ok = b.logical_and(b.ult(b_k_row, k), b.ult(n_col, n));
        let bv = b.var_f32(f0);
        b.if_then(b_ok, |b| {
            let idx = b.iadd(off_b, b.iadd(b.imul(b_k_row, n), n_col));
            let v = b.arena_load(idx);
            b.store(bv, v);
        });
        let bv_val = b.load_f32(bv);
        b.shared_store(tile_b, l_flat, bv_val);

        b.workgroup_barrier();

        b.for_range(u0, u16, u1, |b, p| {
            let ta = b.shared_load(tile_a, b.iadd(b.imul(ly, u16), p));
            let tb = b.shared_load(tile_b, b.iadd(b.imul(lx, u16), p));
            let cur = b.load_f32(acc);
            b.store(acc, b.fadd(cur, b.fmul(ta, tb)));
        });

        b.workgroup_barrier();
    });

    let c_ok = b.logical_and(b.ult(row, m), b.ult(col, k));
    b.if_then(c_ok, |b| {
        let total = b.load_f32(acc);
        let c_idx = b.iadd(off_c, b.iadd(b.imul(row, k), col));
        b.arena_add_assign(c_idx, total);
    });

    b.finalize()
}

// ===========================================================================
// 10..12. Multi-Head Causal Self-Attention Forward & Race-Free Backward
// ===========================================================================

/// 1 thread por `(h, i)` (`gid.x < n_heads * t`).
/// `p0 = off_q, p1 = off_k, p2 = off_v, p3 = off_probs, p4 = off_ctx, p5 = t, p6 = d, p7 = n_heads, p8 = scale_bits`
fn build_causal_mha_fwd() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let gid = b.global_id(0);
    let off_q = b.push_u32(0);
    let off_k = b.push_u32(1);
    let off_v = b.push_u32(2);
    let off_probs = b.push_u32(3);
    let off_ctx = b.push_u32(4);
    let t = b.push_u32(5);
    let d = b.push_u32(6);
    let n_heads = b.push_u32(7);
    let scale = b.push_f32(8);

    let total_hi = b.imul(n_heads, t);
    let cond = b.ult(gid, total_hi);
    b.if_then(cond, |b| {
        let u0 = b.const_u32(0);
        let u1 = b.const_u32(1);
        let f0 = b.const_f32(0.0);
        let f1 = b.const_f32(1.0);
        let f_eps = b.const_f32(1e-12);
        let neg_big = b.const_f32(-1.0e30);

        let h = b.udiv(gid, t);
        let i = b.umod(gid, t);
        let head_dim = b.udiv(d, n_heads);
        let h_off = b.imul(h, head_dim);
        let i_plus_1 = b.iadd(i, u1);
        let prob_row_off = b.iadd(off_probs, b.imul(gid, t));
        let q_base = b.iadd(off_q, b.iadd(b.imul(i, d), h_off));

        let max_s = b.var_f32(neg_big);
        b.for_range(u0, i_plus_1, u1, |b, j| {
            let k_base = b.iadd(off_k, b.iadd(b.imul(j, d), h_off));
            let dot = b.var_f32(f0);
            b.for_range(u0, head_dim, u1, |b, hd| {
                let qv = b.arena_load(b.iadd(q_base, hd));
                let kv = b.arena_load(b.iadd(k_base, hd));
                let cur = b.load_f32(dot);
                b.store(dot, b.fadd(cur, b.fmul(qv, kv)));
            });
            let s = b.fmul(b.load_f32(dot), scale);
            b.arena_store(b.iadd(prob_row_off, j), s);
            let cur_mx = b.load_f32(max_s);
            b.store(max_s, b.fmax(cur_mx, s));
        });

        let mx = b.load_f32(max_s);
        let sum_e = b.var_f32(f0);
        b.for_range(u0, i_plus_1, u1, |b, j| {
            let pj_idx = b.iadd(prob_row_off, j);
            let s = b.arena_load(pj_idx);
            let e = b.fexp(b.fsub(s, mx));
            b.arena_store(pj_idx, e);
            let cur = b.load_f32(sum_e);
            b.store(sum_e, b.fadd(cur, e));
        });

        let inv = b.fdiv(f1, b.fmax(b.load_f32(sum_e), f_eps));
        b.for_range(u0, i_plus_1, u1, |b, j| {
            let pj_idx = b.iadd(prob_row_off, j);
            let p = b.fmul(b.arena_load(pj_idx), inv);
            b.arena_store(pj_idx, p);
        });

        let ctx_base = b.iadd(off_ctx, b.iadd(b.imul(i, d), h_off));
        b.for_range(u0, head_dim, u1, |b, hd| {
            let acc_c = b.var_f32(f0);
            b.for_range(u0, i_plus_1, u1, |b, j| {
                let p = b.arena_load(b.iadd(prob_row_off, j));
                let vv = b.arena_load(b.iadd(off_v, b.iadd(b.iadd(b.imul(j, d), h_off), hd)));
                let cur = b.load_f32(acc_c);
                b.store(acc_c, b.fadd(cur, b.fmul(p, vv)));
            });
            let cv = b.load_f32(acc_c);
            b.arena_store(b.iadd(ctx_base, hd), cv);
        });
    });
    b.finalize()
}

/// Atenção Backward Passo 1 (`gid.x < n_heads * t`): calcula `d_scores[h, i, j]` e `d_q[i, h, hd]`.
/// `p0 = off_k, p1 = off_v, p2 = off_probs, p3 = off_dctx, p4 = off_dscores, p5 = off_dq, p6 = t, p7 = d, p8 = n_heads, p9 = scale_bits`
fn build_causal_mha_bwd_q_ds() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let gid = b.global_id(0);
    let off_k = b.push_u32(0);
    let off_v = b.push_u32(1);
    let off_probs = b.push_u32(2);
    let off_dctx = b.push_u32(3);
    let off_dscores = b.push_u32(4);
    let off_dq = b.push_u32(5);
    let t = b.push_u32(6);
    let d = b.push_u32(7);
    let n_heads = b.push_u32(8);
    let scale = b.push_f32(9);

    let total_hi = b.imul(n_heads, t);
    let cond = b.ult(gid, total_hi);
    b.if_then(cond, |b| {
        let u0 = b.const_u32(0);
        let u1 = b.const_u32(1);
        let f0 = b.const_f32(0.0);

        let h = b.udiv(gid, t);
        let i = b.umod(gid, t);
        let head_dim = b.udiv(d, n_heads);
        let h_off = b.imul(h, head_dim);
        let i_plus_1 = b.iadd(i, u1);
        let prob_row = b.iadd(off_probs, b.imul(gid, t));
        let ds_row = b.iadd(off_dscores, b.imul(gid, t));
        let dctx_base = b.iadd(off_dctx, b.iadd(b.imul(i, d), h_off));

        // Guarda temporariamente `dp[j]` em `ds_row[j]` e acumula `dot_pd`
        let dot_pd = b.var_f32(f0);
        b.for_range(u0, i_plus_1, u1, |b, j| {
            let v_base = b.iadd(off_v, b.iadd(b.imul(j, d), h_off));
            let dp = b.var_f32(f0);
            b.for_range(u0, head_dim, u1, |b, hd| {
                let dov = b.arena_load(b.iadd(dctx_base, hd));
                let vv = b.arena_load(b.iadd(v_base, hd));
                let cur = b.load_f32(dp);
                b.store(dp, b.fadd(cur, b.fmul(dov, vv)));
            });
            let dp_val = b.load_f32(dp);
            b.arena_store(b.iadd(ds_row, j), dp_val);
            let p = b.arena_load(b.iadd(prob_row, j));
            let cur_pd = b.load_f32(dot_pd);
            b.store(dot_pd, b.fadd(cur_pd, b.fmul(p, dp_val)));
        });

        let pd = b.load_f32(dot_pd);
        b.for_range(u0, i_plus_1, u1, |b, j| {
            let p = b.arena_load(b.iadd(prob_row, j));
            let dp_val = b.arena_load(b.iadd(ds_row, j));
            let ds = b.fmul(b.fmul(p, b.fsub(dp_val, pd)), scale);
            b.arena_store(b.iadd(ds_row, j), ds);
        });

        let dq_base = b.iadd(off_dq, b.iadd(b.imul(i, d), h_off));
        b.for_range(u0, head_dim, u1, |b, hd| {
            let acc_q = b.var_f32(f0);
            b.for_range(u0, i_plus_1, u1, |b, j| {
                let ds = b.arena_load(b.iadd(ds_row, j));
                let kv = b.arena_load(b.iadd(off_k, b.iadd(b.iadd(b.imul(j, d), h_off), hd)));
                let cur = b.load_f32(acc_q);
                b.store(acc_q, b.fadd(cur, b.fmul(ds, kv)));
            });
            let dqv = b.load_f32(acc_q);
            b.arena_store(b.iadd(dq_base, hd), dqv);
        });
    });
    b.finalize()
}

/// Atenção Backward Passo 2 (`gid.x < t * d`): reduz `i in j..t` calculando `d_k[j, col]` e `d_v[j, col]` sem atomics!
/// `p0 = off_q, p1 = off_probs, p2 = off_dctx, p3 = off_dscores, p4 = off_dk, p5 = off_dv, p6 = t, p7 = d, p8 = n_heads`
fn build_causal_mha_bwd_kv() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let gid = b.global_id(0);
    let off_q = b.push_u32(0);
    let off_probs = b.push_u32(1);
    let off_dctx = b.push_u32(2);
    let off_dscores = b.push_u32(3);
    let off_dk = b.push_u32(4);
    let off_dv = b.push_u32(5);
    let t = b.push_u32(6);
    let d = b.push_u32(7);
    let n_heads = b.push_u32(8);

    let td = b.imul(t, d);
    let cond = b.ult(gid, td);
    b.if_then(cond, |b| {
        let u1 = b.const_u32(1);
        let f0 = b.const_f32(0.0);

        let j = b.udiv(gid, d);
        let col = b.umod(gid, d);
        let head_dim = b.udiv(d, n_heads);
        let h = b.udiv(col, head_dim);
        let ht = b.imul(h, t);

        let sum_dk = b.var_f32(f0);
        let sum_dv = b.var_f32(f0);

        b.for_range(j, t, u1, |b, i| {
            let hi_row = b.imul(b.iadd(ht, i), t);
            let p_ij = b.arena_load(b.iadd(off_probs, b.iadd(hi_row, j)));
            let ds_ij = b.arena_load(b.iadd(off_dscores, b.iadd(hi_row, j)));
            let ic = b.iadd(b.imul(i, d), col);
            let qv = b.arena_load(b.iadd(off_q, ic));
            let dov = b.arena_load(b.iadd(off_dctx, ic));

            let cdk = b.load_f32(sum_dk);
            b.store(sum_dk, b.fadd(cdk, b.fmul(ds_ij, qv)));
            let cdv = b.load_f32(sum_dv);
            b.store(sum_dv, b.fadd(cdv, b.fmul(p_ij, dov)));
        });

        let dk_val = b.load_f32(sum_dk);
        let dv_val = b.load_f32(sum_dv);
        b.arena_store(b.iadd(off_dk, gid), dk_val);
        b.arena_store(b.iadd(off_dv, gid), dv_val);
    });
    b.finalize()
}

// ===========================================================================
// 13..16. MLP (Bias + GELU + Residuais + Gradientes de Bias)
// ===========================================================================

/// `p0 = off_hpre, p1 = off_b1, p2 = off_hact, p3 = total_elems (t * ff), p4 = ff`
fn build_bias_gelu_fwd() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_hpre = b.push_u32(0);
    let off_b1 = b.push_u32(1);
    let off_hact = b.push_u32(2);
    let total = b.push_u32(3);
    let ff = b.push_u32(4);

    let cond = b.ult(gid, total);
    b.if_then(cond, |b| {
        let col = b.umod(gid, ff);
        let hp_idx = b.iadd(off_hpre, gid);
        let z = b.fadd(b.arena_load(hp_idx), b.arena_load(b.iadd(off_b1, col)));
        b.arena_store(hp_idx, z);
        let g = emit_gelu_f32(b, z);
        b.arena_store(b.iadd(off_hact, gid), g);
    });
    b.finalize()
}

/// `p0 = off_dhact, p1 = off_hpre, p2 = off_dhpre, p3 = total_elems (t * ff)`
fn build_gelu_bwd() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_dhact = b.push_u32(0);
    let off_hpre = b.push_u32(1);
    let off_dhpre = b.push_u32(2);
    let total = b.push_u32(3);

    let cond = b.ult(gid, total);
    b.if_then(cond, |b| {
        let dha = b.arena_load(b.iadd(off_dhact, gid));
        let z = b.arena_load(b.iadd(off_hpre, gid));
        let gg = emit_gelu_grad_f32(b, z);
        let dhp = b.fmul(dha, gg);
        b.arena_store(b.iadd(off_dhpre, gid), dhp);
    });
    b.finalize()
}

/// `p0 = off_xmid, p1 = off_mlp, p2 = off_b2, p3 = off_xout, p4 = total_elems (t * d), p5 = d`
fn build_bias_res_add_fwd() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);
    let off_xmid = b.push_u32(0);
    let off_mlp = b.push_u32(1);
    let off_b2 = b.push_u32(2);
    let off_xout = b.push_u32(3);
    let total = b.push_u32(4);
    let d = b.push_u32(5);

    let cond = b.ult(gid, total);
    b.if_then(cond, |b| {
        let col = b.umod(gid, d);
        let xm = b.arena_load(b.iadd(off_xmid, gid));
        let mo = b.arena_load(b.iadd(off_mlp, gid));
        let b2 = b.arena_load(b.iadd(off_b2, col));
        let res = b.fadd(b.fadd(xm, mo), b2);
        b.arena_store(b.iadd(off_xout, gid), res);
    });
    b.finalize()
}

/// `p0 = off_dmat, p1 = off_dbias, p2 = t, p3 = cols` (`gid.x < cols`)
fn build_bias_grad_sum() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let col = b.global_id(0);
    let off_dmat = b.push_u32(0);
    let off_dbias = b.push_u32(1);
    let t = b.push_u32(2);
    let cols = b.push_u32(3);

    let cond = b.ult(col, cols);
    b.if_then(cond, |b| {
        let u0 = b.const_u32(0);
        let u1 = b.const_u32(1);
        let f0 = b.const_f32(0.0);
        let sum = b.var_f32(f0);
        b.for_range(u0, t, u1, |b, pos| {
            let v = b.arena_load(b.iadd(off_dmat, b.iadd(b.imul(pos, cols), col)));
            let cur = b.load_f32(sum);
            b.store(sum, b.fadd(cur, v));
        });
        let total = b.load_f32(sum);
        b.arena_add_assign(b.iadd(off_dbias, col), total);
    });
    b.finalize()
}

// ===========================================================================
// 17..21. Cross-Entropy Log-Sum-Exp, Reduções, Norma de Gradiente e AdamW
// ===========================================================================

/// 1 Workgroup (`64` threads) por posição `pos = wgid.x < t`.
/// Calcula perda exata `log-sum-exp` por token em `row_losses[pos]` e `d_logits[pos, 0..v]`
/// (quando `write_dlogits == 1`).
/// `p0 = off_logits, p1 = off_dlogits, p2 = off_row_losses, p3 = v, p4 = pad_id, p5 = io_tgt_off, p6 = write_dlogits`
fn build_cross_entropy_fwd_bwd() -> Vec<u32> {
    let b = SpvBuilder::new(64, 1, 1);
    let shared = b.decl_shared_f32_array(64);
    let pos = b.workgroup_id(0);
    let lid = b.local_id(0);

    let off_logits = b.push_u32(0);
    let off_dlogits = b.push_u32(1);
    let off_losses = b.push_u32(2);
    let v = b.push_u32(3);
    let pad_id = b.push_u32(4);
    let io_tgt = b.push_u32(5);
    let write_dl = b.push_u32(6);

    let u0 = b.const_u32(0);
    let u1 = b.const_u32(1);
    let c64 = b.const_u32(64);
    let f0 = b.const_f32(0.0);
    let f1 = b.const_f32(1.0);
    let f_min_sum = b.const_f32(1e-30);
    let neg_big = b.const_f32(-1.0e30);

    let tgt = b.io_load_u32(b.iadd(io_tgt, pos));
    let is_pad = b.ieq(tgt, pad_id);
    let out_of_vocab = b.logical_or(b.ieq(tgt, v), b.ult(v, tgt));
    let skip = b.logical_or(is_pad, out_of_vocab);
    let row_off = b.imul(pos, v);

    // Como `skip` depende apenas de `pos = wgid.x`, a condição é 100% uniforme no Workgroup!
    b.if_else(
        skip,
        |b| {
            let is_l0 = b.ieq(lid, u0);
            b.if_then(is_l0, |b| {
                b.arena_store(b.iadd(off_losses, pos), f0);
            });
            let do_write = b.ieq(write_dl, u1);
            b.if_then(do_write, |b| {
                b.for_range(lid, v, c64, |b, j| {
                    b.arena_store(b.iadd(off_dlogits, b.iadd(row_off, j)), f0);
                });
            });
        },
        |b| {
            // 1. Reduz `max_l` na linha
            let loc_max = b.var_f32(neg_big);
            b.for_range(lid, v, c64, |b, j| {
                let lv = b.arena_load(b.iadd(off_logits, b.iadd(row_off, j)));
                let cur = b.load_f32(loc_max);
                b.store(loc_max, b.fmax(cur, lv));
            });
            let lm = b.load_f32(loc_max);
            b.shared_store(shared, lid, lm);
            emit_wg_tree_max(b, shared, lid, 64);
            let max_l = b.shared_load(shared, u0);

            // 2. Reduz `sum_e = sum_j exp(row[j] - max_l)`
            let loc_sum = b.var_f32(f0);
            b.for_range(lid, v, c64, |b, j| {
                let lv = b.arena_load(b.iadd(off_logits, b.iadd(row_off, j)));
                let e = b.fexp(b.fsub(lv, max_l));
                let cur = b.load_f32(loc_sum);
                b.store(loc_sum, b.fadd(cur, e));
            });
            let ls = b.load_f32(loc_sum);
            b.shared_store(shared, lid, ls);
            emit_wg_tree_sum(b, shared, lid, 64);
            let safe_sum = b.fmax(b.shared_load(shared, u0), f_min_sum);

            // 3. Lane 0 grava a loss exata da posição
            let is_l0 = b.ieq(lid, u0);
            b.if_then(is_l0, |b| {
                let l_tgt = b.arena_load(b.iadd(off_logits, b.iadd(row_off, tgt)));
                let log_prob = b.fsub(b.fsub(l_tgt, max_l), b.flog(safe_sum));
                let loss = b.fsub(f0, log_prob);
                b.arena_store(b.iadd(off_losses, pos), loss);
            });

            // 4. Escreve `d_logits` se solicitado
            let do_write = b.ieq(write_dl, u1);
            b.if_then(do_write, |b| {
                let inv = b.fdiv(f1, safe_sum);
                b.for_range(lid, v, c64, |b, j| {
                    let lv = b.arena_load(b.iadd(off_logits, b.iadd(row_off, j)));
                    let prob = b.fmul(b.fexp(b.fsub(lv, max_l)), inv);
                    let is_tgt = b.ieq(j, tgt);
                    let sub_one = b.fsub(prob, f1);
                    let grad = b.select_f32(is_tgt, sub_one, prob);
                    b.arena_store(b.iadd(off_dlogits, b.iadd(row_off, j)), grad);
                });
            });
        },
    );

    b.finalize()
}

/// 1 Workgroup (`256` threads) reduzindo `arena[off_src .. off_src + len]` e gravando em `io[io_out_idx]`.
/// `p0 = off_src, p1 = len, p2 = io_out_idx, p3 = accumulate_io (0 = sobrescreve, 1 = soma ao valor atual)`
fn build_reduce_sum_f32_to_io() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let shared = b.decl_shared_f32_array(256);
    let lid = b.local_id(0);
    let off_src = b.push_u32(0);
    let len = b.push_u32(1);
    let io_out = b.push_u32(2);
    let accum = b.push_u32(3);

    let u0 = b.const_u32(0);
    let u1 = b.const_u32(1);
    let c256 = b.const_u32(256);
    let f0 = b.const_f32(0.0);

    let acc = b.var_f32(f0);
    b.for_range(lid, len, c256, |b, idx| {
        let v = b.arena_load(b.iadd(off_src, idx));
        let cur = b.load_f32(acc);
        b.store(acc, b.fadd(cur, v));
    });
    let part = b.load_f32(acc);
    b.shared_store(shared, lid, part);
    emit_wg_tree_sum(&b, shared, lid, 256);

    let is_l0 = b.ieq(lid, u0);
    b.if_then(is_l0, |b| {
        let sum = b.shared_load(shared, u0);
        let do_acc = b.ieq(accum, u1);
        b.if_else(
            do_acc,
            |b| {
                let prev_bits = b.io_load_u32(io_out);
                let prev_f = b.alloc_id();
                b.emit_body(OP_BITCAST, &[b.ty_f32, prev_f, prev_bits]);
                let tot = b.fadd(prev_f, sum);
                b.io_store_f32_as_bits(io_out, tot);
            },
            |b| {
                b.io_store_f32_as_bits(io_out, sum);
            },
        );
    });

    b.finalize()
}

/// Escala `grads[0..len] *= scale_factor` e grava a soma parcial de quadrados de cada Workgroup (`256` threads)
/// em `arena[off_partial + wgid.x]`.
/// `p0 = off_grads, p1 = len, p2 = off_partial, p3 = scale_bits`
fn build_scale_and_grad_sq() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let shared = b.decl_shared_f32_array(256);
    let gid = b.global_id(0);
    let lid = b.local_id(0);
    let wgid = b.workgroup_id(0);

    let off_grads = b.push_u32(0);
    let len = b.push_u32(1);
    let off_partial = b.push_u32(2);
    let scale = b.push_f32(3);

    let u0 = b.const_u32(0);
    let f0 = b.const_f32(0.0);

    let sq = b.var_f32(f0);
    let in_bounds = b.ult(gid, len);
    b.if_then(in_bounds, |b| {
        let idx = b.iadd(off_grads, gid);
        let g = b.fmul(b.arena_load(idx), scale);
        b.arena_store(idx, g);
        b.store(sq, b.fmul(g, g));
    });

    let my_sq = b.load_f32(sq);
    b.shared_store(shared, lid, my_sq);
    emit_wg_tree_sum(&b, shared, lid, 256);

    let is_l0 = b.ieq(lid, u0);
    b.if_then(is_l0, |b| {
        let wg_sum = b.shared_load(shared, u0);
        b.arena_store(b.iadd(off_partial, wgid), wg_sum);
    });

    b.finalize()
}

/// Atualização AdamW em VRAM sobre uma fatia contígua `[0..len]` de parâmetros:
/// `p0 = off_w, p1 = off_g, p2 = off_m, p3 = off_v, p4 = len, p5 = lr_bits, p6 = wd_eff_bits, p7 = bc1_bits, p8 = bc2_bits, p9 = clip_scale_bits`
fn build_adamw_step() -> Vec<u32> {
    let b = SpvBuilder::new(256, 1, 1);
    let gid = b.global_id(0);

    let off_w = b.push_u32(0);
    let off_g = b.push_u32(1);
    let off_m = b.push_u32(2);
    let off_v = b.push_u32(3);
    let len = b.push_u32(4);
    let lr = b.push_f32(5);
    let wd_eff = b.push_f32(6); // `lr * weight_decay` quando `apply_decay`, senão `0.0`
    let bc1 = b.push_f32(7);
    let bc2 = b.push_f32(8);
    let clip_scale = b.push_f32(9);

    let cond = b.ult(gid, len);
    b.if_then(cond, |b| {
        let beta1 = b.const_f32(0.9);
        let one_m_b1 = b.const_f32(0.1);
        let beta2 = b.const_f32(0.999);
        let one_m_b2 = b.const_f32(0.001);
        let eps = b.const_f32(1e-8);

        let wi = b.iadd(off_w, gid);
        let gi = b.iadd(off_g, gid);
        let mi = b.iadd(off_m, gid);
        let vi = b.iadd(off_v, gid);

        let g = b.fmul(b.arena_load(gi), clip_scale);
        b.arena_store(gi, g);

        let m_new = b.fadd(b.fmul(beta1, b.arena_load(mi)), b.fmul(one_m_b1, g));
        let v_new = b.fadd(
            b.fmul(beta2, b.arena_load(vi)),
            b.fmul(one_m_b2, b.fmul(g, g)),
        );
        b.arena_store(mi, m_new);
        b.arena_store(vi, v_new);

        let m_hat = b.fdiv(m_new, bc1);
        let v_hat = b.fdiv(v_new, bc2);

        let p_old = b.arena_load(wi);
        let p_decayed = b.fsub(p_old, b.fmul(wd_eff, p_old));
        let denom = b.fadd(b.fsqrt(v_hat), eps);
        let step_delta = b.fmul(lr, b.fdiv(m_hat, denom));
        let p_new = b.fsub(p_decayed, step_delta);
        b.arena_store(wi, p_new);
    });

    b.finalize()
}

/// Verifica invariantes básicas de cabeçalho e palavras em um módulo SPIR-V gerado.
#[cfg(test)]
pub fn validate_spirv_header(words: &[u32]) -> Result<(), String> {
    if words.len() < 10 {
        return Err("Módulo SPIR-V curto demais.".to_string());
    }
    if words[0] != SPV_MAGIC {
        return Err(format!("Magic SPIR-V inválido: {:08x}", words[0]));
    }
    if words[1] != SPV_VERSION_1_3 {
        return Err(format!("Versão SPIR-V inesperada: {:08x}", words[1]));
    }
    if words[3] == 0 {
        return Err("Bound SPIR-V zero.".to_string());
    }
    let mut idx = 5usize;
    while idx < words.len() {
        let wc = (words[idx] >> 16) as usize;
        let op = (words[idx] & 0xFFFF) as u16;
        if wc == 0 || idx + wc > words.len() {
            return Err(format!(
                "Instrução SPIR-V malformada em word {}: op={}, wc={}, len={}",
                idx,
                op,
                wc,
                words.len()
            ));
        }
        idx += wc;
    }
    Ok(())
}
