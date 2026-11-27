//! Backend GPU Modular via Vulkan Compute Dinâmico (`SM-006` — Etapa 7).
//!
//! Projetado especificamente para a **AMD Radeon RX 7600 (8 GB GDDR6, RDNA 3)** sob Windows
//! e Linux sem depender de ROCm, `VULKAN_SDK`, `glslc` ou crates externas:
//! 1. Carrega `vulkan-1.dll` (Windows) ou `libvulkan.so.1` (Linux) dinamicamente em runtime;
//!    se ausente (ex.: PCs modestos de 8 GB RAM sem GPU dedicada), degrada limpamente para CPU.
//! 2. Aloca uma única **Arena `DEVICE_LOCAL` em VRAM** contendo pesos, gradientes, momentos `m`/`v`
//!    do AdamW, ativações de camada e tensores temporários — eliminando o gargalo `PCIe 3.0 x8`
//!    (~5,8 GB/s) durante os passos de treinamento.
//! 3. Submete todo o grafo de `forward + cross_entropy + backward` de uma amostra em um único
//!    `VkCommandBuffer` sincronizado por barreiras internas de compute em VRAM.

use super::spirv::{NeuralSpirvKernels, PUSH_CONSTANT_BYTES, PUSH_CONSTANT_WORDS};
use super::{AdamWState, ServerMindTransformer, TransformerConfig};
use serde::{Deserialize, Serialize};
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

/// Limiar calibrado empiricamente no i5-9600K + RX 7600 para que `ComputeDevice::Auto`
/// selecione automaticamente a GPU discreta Vulkan em modelos reais (`>= 250K` params;
/// em `1.45M` params a RX 7600 já entrega `4.01x` sobre CPU AVX2+FMA), mantendo modelos
/// microscópicos de testes unitários na CPU.
pub const GPU_AUTO_MIN_PARAMS: usize = 250_000;

/// Seleção explícita ou automática do backend computacional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDevice {
    Auto,
    Cpu,
    Gpu,
}

impl Default for ComputeDevice {
    fn default() -> Self {
        Self::Auto
    }
}

impl ComputeDevice {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "vulkan" | "rx7600" => Ok(Self::Gpu),
            other => Err(format!(
                "Valor inválido '{}' para '--device': use 'auto', 'cpu' ou 'gpu'.",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Metadados reais do dispositivo físico Vulkan detectado em runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanAdapterInfo {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub is_discrete_gpu: bool,
    pub api_version_str: String,
    pub device_local_vram_mb: u64,
    pub max_storage_buffer_range: u64,
}

// ===========================================================================
// Definições Mínimas C ABI do Vulkan 1.1 Compute
// ===========================================================================

type VkInstance = *mut c_void;
type VkPhysicalDevice = *mut c_void;
type VkDevice = *mut c_void;
type VkQueue = *mut c_void;
type VkCommandBuffer = *mut c_void;

type VkBuffer = u64;
type VkDeviceMemory = u64;
type VkShaderModule = u64;
type VkDescriptorSetLayout = u64;
type VkPipelineLayout = u64;
type VkPipeline = u64;
type VkDescriptorPool = u64;
type VkDescriptorSet = u64;
type VkCommandPool = u64;
type VkFence = u64;
type VkResult = i32;

const VK_SUCCESS: VkResult = 0;
const VK_API_VERSION_1_1: u32 = (1 << 22) | (1 << 12);
const VK_WHOLE_SIZE: u64 = !0u64;

const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;

const VK_QUEUE_COMPUTE_BIT: u32 = 0x0000_0002;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0020;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x0000_0001;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x0000_0002;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0020;

const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0000_0002;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0000_0004;
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;

const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x0000_0002;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x0000_0001;

const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x0000_0800;
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x0000_1000;
const VK_PIPELINE_STAGE_HOST_BIT: u32 = 0x0000_4000;

const VK_ACCESS_SHADER_READ_BIT: u32 = 0x0000_0020;
const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x0000_0040;
const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x0000_0800;
const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x0000_1000;
const VK_ACCESS_HOST_READ_BIT: u32 = 0x0000_2000;

// VkStructureType
const VK_STYPE_APPLICATION_INFO: u32 = 0;
const VK_STYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STYPE_SUBMIT_INFO: u32 = 4;
const VK_STYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STYPE_MEMORY_BARRIER: u32 = 46;

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const i8,
    application_version: u32,
    p_engine_name: *const i8,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const i8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const i8,
}

/// Buffer alinhado seguro (`1024` bytes >= `sizeof(VkPhysicalDeviceProperties)` = `824` bytes em x64)
/// para ler os campos iniciais sem risco de desalinhamento interno de `VkPhysicalDeviceLimits`.
#[repr(C, align(8))]
struct RawPhysicalDeviceProperties {
    bytes: [u8; 1024],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const i8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const i8,
    p_enabled_features: *const c_void,
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

#[repr(C)]
struct VkBufferCopy {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    p_code: *const u32,
}

#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    p_immutable_samplers: *const c_void,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
struct VkPushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    p_name: *const i8,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
struct VkDescriptorPoolSize {
    ty: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const c_void,
    p_buffer_info: *const VkDescriptorBufferInfo,
    p_texel_buffer_view: *const u64,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct VkMemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: u32,
    dst_access_mask: u32,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const u64,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

// ===========================================================================
// Carregador Dinâmico da Biblioteca Vulkan (`vulkan-1.dll` / `libvulkan.so.1`)
// ===========================================================================

type PfnVoid = Option<unsafe extern "system" fn()>;
type PfnGetInstanceProcAddr = unsafe extern "system" fn(VkInstance, *const i8) -> PfnVoid;

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lp_lib_file_name: *const i8) -> *mut c_void;
    fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const i8) -> *mut c_void;
}

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
}

unsafe fn load_vulkan_entry() -> Option<PfnGetInstanceProcAddr> {
    #[cfg(target_os = "windows")]
    {
        let lib = LoadLibraryA(b"vulkan-1.dll\0".as_ptr() as *const i8);
        if lib.is_null() {
            return None;
        }
        let sym = GetProcAddress(lib, b"vkGetInstanceProcAddr\0".as_ptr() as *const i8);
        if sym.is_null() {
            return None;
        }
        Some(std::mem::transmute::<*mut c_void, PfnGetInstanceProcAddr>(sym))
    }
    #[cfg(unix)]
    {
        const RTLD_NOW: i32 = 2;
        for candidate in [
            b"libvulkan.so.1\0".as_slice(),
            b"libvulkan.so\0".as_slice(),
        ] {
            let lib = dlopen(candidate.as_ptr() as *const i8, RTLD_NOW);
            if !lib.is_null() {
                let sym = dlsym(lib, b"vkGetInstanceProcAddr\0".as_ptr() as *const i8);
                if !sym.is_null() {
                    return Some(std::mem::transmute::<*mut c_void, PfnGetInstanceProcAddr>(sym));
                }
            }
        }
        None
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        None
    }
}

macro_rules! load_vk_fn {
    ($get_proc:expr, $inst:expr, $name:literal, $ty:ty) => {{
        let ptr = ($get_proc)($inst, concat!($name, "\0").as_ptr() as *const i8);
        let Some(f) = ptr else {
            return Err(format!(" Símbolo Vulkan ausente: {}", $name));
        };
        std::mem::transmute::<unsafe extern "system" fn(), $ty>(f)
    }};
}

struct VkFns {
    destroy_instance: unsafe extern "system" fn(VkInstance, *const c_void),
    enumerate_physical_devices:
        unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult,
    get_physical_device_properties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut RawPhysicalDeviceProperties),
    get_physical_device_memory_properties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties),
    get_physical_device_queue_family_properties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties),
    create_device: unsafe extern "system" fn(
        VkPhysicalDevice,
        *const VkDeviceCreateInfo,
        *const c_void,
        *mut VkDevice,
    ) -> VkResult,
    destroy_device: unsafe extern "system" fn(VkDevice, *const c_void),
    get_device_queue: unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue),
    create_buffer: unsafe extern "system" fn(
        VkDevice,
        *const VkBufferCreateInfo,
        *const c_void,
        *mut VkBuffer,
    ) -> VkResult,
    destroy_buffer: unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void),
    get_buffer_memory_requirements:
        unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements),
    allocate_memory: unsafe extern "system" fn(
        VkDevice,
        *const VkMemoryAllocateInfo,
        *const c_void,
        *mut VkDeviceMemory,
    ) -> VkResult,
    free_memory: unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void),
    bind_buffer_memory:
        unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> VkResult,
    map_memory: unsafe extern "system" fn(
        VkDevice,
        VkDeviceMemory,
        u64,
        u64,
        u32,
        *mut *mut c_void,
    ) -> VkResult,
    unmap_memory: unsafe extern "system" fn(VkDevice, VkDeviceMemory),
    create_shader_module: unsafe extern "system" fn(
        VkDevice,
        *const VkShaderModuleCreateInfo,
        *const c_void,
        *mut VkShaderModule,
    ) -> VkResult,
    destroy_shader_module: unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void),
    create_descriptor_set_layout: unsafe extern "system" fn(
        VkDevice,
        *const VkDescriptorSetLayoutCreateInfo,
        *const c_void,
        *mut VkDescriptorSetLayout,
    ) -> VkResult,
    destroy_descriptor_set_layout:
        unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void),
    create_pipeline_layout: unsafe extern "system" fn(
        VkDevice,
        *const VkPipelineLayoutCreateInfo,
        *const c_void,
        *mut VkPipelineLayout,
    ) -> VkResult,
    destroy_pipeline_layout: unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void),
    create_compute_pipelines: unsafe extern "system" fn(
        VkDevice,
         u64,
        u32,
        *const VkComputePipelineCreateInfo,
        *const c_void,
        *mut VkPipeline,
    ) -> VkResult,
    destroy_pipeline: unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void),
    create_descriptor_pool: unsafe extern "system" fn(
        VkDevice,
        *const VkDescriptorPoolCreateInfo,
        *const c_void,
        *mut VkDescriptorPool,
    ) -> VkResult,
    destroy_descriptor_pool: unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void),
    allocate_descriptor_sets: unsafe extern "system" fn(
        VkDevice,
        *const VkDescriptorSetAllocateInfo,
        *mut VkDescriptorSet,
    ) -> VkResult,
    update_descriptor_sets:
        unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void),
    create_command_pool: unsafe extern "system" fn(
        VkDevice,
        *const VkCommandPoolCreateInfo,
        *const c_void,
        *mut VkCommandPool,
    ) -> VkResult,
    destroy_command_pool: unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void),
    allocate_command_buffers: unsafe extern "system" fn(
        VkDevice,
        *const VkCommandBufferAllocateInfo,
        *mut VkCommandBuffer,
    ) -> VkResult,
    reset_command_buffer: unsafe extern "system" fn(VkCommandBuffer, u32) -> VkResult,
    begin_command_buffer:
        unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult,
    end_command_buffer: unsafe extern "system" fn(VkCommandBuffer) -> VkResult,
    cmd_bind_pipeline: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline),
    cmd_bind_descriptor_sets: unsafe extern "system" fn(
        VkCommandBuffer,
        u32,
        VkPipelineLayout,
        u32,
        u32,
        *const VkDescriptorSet,
        u32,
        *const u32,
    ),
    cmd_push_constants:
        unsafe extern "system" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const c_void),
    cmd_dispatch: unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32),
    cmd_copy_buffer:
        unsafe extern "system" fn(VkCommandBuffer, VkBuffer, VkBuffer, u32, *const VkBufferCopy),
    cmd_pipeline_barrier: unsafe extern "system" fn(
        VkCommandBuffer,
        u32,
        u32,
        u32,
        u32,
        *const VkMemoryBarrier,
        u32,
        *const c_void,
        u32,
        *const c_void,
    ),
    create_fence: unsafe extern "system" fn(
        VkDevice,
        *const VkFenceCreateInfo,
        *const c_void,
        *mut VkFence,
    ) -> VkResult,
    destroy_fence: unsafe extern "system" fn(VkDevice, VkFence, *const c_void),
    wait_for_fences:
        unsafe extern "system" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult,
    reset_fences: unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> VkResult,
    queue_submit:
        unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult,
}

struct CompiledPipelines {
    fill_f32: VkPipeline,
    vec_add_inplace: VkPipeline,
    embed_fwd: VkPipeline,
    embed_bwd: VkPipeline,
    rmsnorm_fwd: VkPipeline,
    rmsnorm_bwd_dx: VkPipeline,
    rmsnorm_bwd_dgamma: VkPipeline,
    matmul_nn: VkPipeline,
    matmul_tn: VkPipeline,
    matmul_nt: VkPipeline,
    causal_mha_fwd: VkPipeline,
    causal_mha_bwd_q_ds: VkPipeline,
    causal_mha_bwd_kv: VkPipeline,
    bias_gelu_fwd: VkPipeline,
    gelu_bwd: VkPipeline,
    bias_res_add_fwd: VkPipeline,
    bias_grad_sum: VkPipeline,
    cross_entropy_fwd_bwd: VkPipeline,
    reduce_sum_f32_to_io: VkPipeline,
    scale_and_grad_sq: VkPipeline,
    adamw_step: VkPipeline,
}

impl CompiledPipelines {
    fn all_handles(&self) -> [VkPipeline; 21] {
        [
            self.fill_f32,
            self.vec_add_inplace,
            self.embed_fwd,
            self.embed_bwd,
            self.rmsnorm_fwd,
            self.rmsnorm_bwd_dx,
            self.rmsnorm_bwd_dgamma,
            self.matmul_nn,
            self.matmul_tn,
            self.matmul_nt,
            self.causal_mha_fwd,
            self.causal_mha_bwd_q_ds,
            self.causal_mha_bwd_kv,
            self.bias_gelu_fwd,
            self.gelu_bwd,
            self.bias_res_add_fwd,
            self.bias_grad_sum,
            self.cross_entropy_fwd_bwd,
            self.reduce_sum_f32_to_io,
            self.scale_and_grad_sq,
            self.adamw_step,
        ]
    }
}

/// Dispositivo Vulkan Compute inicializado e pronto para criar sessões de treinamento/inferência.
pub struct VulkanDevice {
    pub info: VulkanAdapterInfo,
    instance: VkInstance,
    device: VkDevice,
    queue_family: u32,
    queue_lock: Mutex<VkQueue>,
    mem_props: VkPhysicalDeviceMemoryProperties,
    desc_set_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipelines: CompiledPipelines,
    fns: VkFns,
}

// SAFETY: Todos os recursos mutáveis de fila são protegidos por `queue_lock: Mutex<VkQueue>`,
// conforme exigência de sincronização externa de `VkQueue` na especificação Vulkan.
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            for p in self.pipelines.all_handles() {
                if p != 0 {
                    (self.fns.destroy_pipeline)(self.device, p, ptr::null());
                }
            }
            if self.pipeline_layout != 0 {
                (self.fns.destroy_pipeline_layout)(self.device, self.pipeline_layout, ptr::null());
            }
            if self.desc_set_layout != 0 {
                (self.fns.destroy_descriptor_set_layout)(
                    self.device,
                    self.desc_set_layout,
                    ptr::null(),
                );
            }
            if !self.device.is_null() {
                (self.fns.destroy_device)(self.device, ptr::null());
            }
            if !self.instance.is_null() {
                (self.fns.destroy_instance)(self.instance, ptr::null());
            }
        }
    }
}

impl VulkanDevice {
    /// Retorna a instância compartilhada da GPU Vulkan local (se disponível e funcional).
    pub fn global() -> Result<Arc<Self>, String> {
        static GLOBAL_DEV: OnceLock<Result<Arc<VulkanDevice>, String>> = OnceLock::new();
        GLOBAL_DEV
            .get_or_init(|| Self::init_best_device().map(Arc::new))
            .clone()
    }

    pub fn try_adapter_info() -> Option<VulkanAdapterInfo> {
        Self::global().ok().map(|d| d.info.clone())
    }

    fn init_best_device() -> Result<Self, String> {
        unsafe {
            let get_proc = load_vulkan_entry()
                .ok_or_else(|| "Runtime Vulkan (vulkan-1.dll / libvulkan.so.1) não encontrado.".to_string())?;

            let create_instance = load_vk_fn!(
                get_proc,
                ptr::null_mut(),
                "vkCreateInstance",
                unsafe extern "system" fn(
                    *const VkInstanceCreateInfo,
                    *const c_void,
                    *mut VkInstance,
                ) -> VkResult
            );

            let app_info = VkApplicationInfo {
                s_type: VK_STYPE_APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: b"ServerMindNeuralEngine\0".as_ptr() as *const i8,
                application_version: 1,
                p_engine_name: b"ServerMind\0".as_ptr() as *const i8,
                engine_version: 1,
                api_version: VK_API_VERSION_1_1,
            };
            let inst_ci = VkInstanceCreateInfo {
                s_type: VK_STYPE_INSTANCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                p_application_info: &app_info,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
            };

            let mut instance: VkInstance = ptr::null_mut();
            let res = create_instance(&inst_ci, ptr::null(), &mut instance);
            if res != VK_SUCCESS || instance.is_null() {
                return Err(format!("vkCreateInstance falhou com código {}.", res));
            }

            let fns = VkFns {
                destroy_instance: load_vk_fn!(get_proc, instance, "vkDestroyInstance", _),
                enumerate_physical_devices: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkEnumeratePhysicalDevices",
                    _
                ),
                get_physical_device_properties: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkGetPhysicalDeviceProperties",
                    _
                ),
                get_physical_device_memory_properties: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkGetPhysicalDeviceMemoryProperties",
                    _
                ),
                get_physical_device_queue_family_properties: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkGetPhysicalDeviceQueueFamilyProperties",
                    _
                ),
                create_device: load_vk_fn!(get_proc, instance, "vkCreateDevice", _),
                destroy_device: load_vk_fn!(get_proc, instance, "vkDestroyDevice", _),
                get_device_queue: load_vk_fn!(get_proc, instance, "vkGetDeviceQueue", _),
                create_buffer: load_vk_fn!(get_proc, instance, "vkCreateBuffer", _),
                destroy_buffer: load_vk_fn!(get_proc, instance, "vkDestroyBuffer", _),
                get_buffer_memory_requirements: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkGetBufferMemoryRequirements",
                    _
                ),
                allocate_memory: load_vk_fn!(get_proc, instance, "vkAllocateMemory", _),
                free_memory: load_vk_fn!(get_proc, instance, "vkFreeMemory", _),
                bind_buffer_memory: load_vk_fn!(get_proc, instance, "vkBindBufferMemory", _),
                map_memory: load_vk_fn!(get_proc, instance, "vkMapMemory", _),
                unmap_memory: load_vk_fn!(get_proc, instance, "vkUnmapMemory", _),
                create_shader_module: load_vk_fn!(get_proc, instance, "vkCreateShaderModule", _),
                destroy_shader_module: load_vk_fn!(get_proc, instance, "vkDestroyShaderModule", _),
                create_descriptor_set_layout: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkCreateDescriptorSetLayout",
                    _
                ),
                destroy_descriptor_set_layout: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkDestroyDescriptorSetLayout",
                    _
                ),
                create_pipeline_layout: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkCreatePipelineLayout",
                    _
                ),
                destroy_pipeline_layout: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkDestroyPipelineLayout",
                    _
                ),
                create_compute_pipelines: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkCreateComputePipelines",
                    _
                ),
                destroy_pipeline: load_vk_fn!(get_proc, instance, "vkDestroyPipeline", _),
                create_descriptor_pool: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkCreateDescriptorPool",
                    _
                ),
                destroy_descriptor_pool: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkDestroyDescriptorPool",
                    _
                ),
                allocate_descriptor_sets: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkAllocateDescriptorSets",
                    _
                ),
                update_descriptor_sets: load_vk_fn!(get_proc, instance, "vkUpdateDescriptorSets", _),
                create_command_pool: load_vk_fn!(get_proc, instance, "vkCreateCommandPool", _),
                destroy_command_pool: load_vk_fn!(get_proc, instance, "vkDestroyCommandPool", _),
                allocate_command_buffers: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkAllocateCommandBuffers",
                    _
                ),
                reset_command_buffer: load_vk_fn!(get_proc, instance, "vkResetCommandBuffer", _),
                begin_command_buffer: load_vk_fn!(get_proc, instance, "vkBeginCommandBuffer", _),
                end_command_buffer: load_vk_fn!(get_proc, instance, "vkEndCommandBuffer", _),
                cmd_bind_pipeline: load_vk_fn!(get_proc, instance, "vkCmdBindPipeline", _),
                cmd_bind_descriptor_sets: load_vk_fn!(
                    get_proc,
                    instance,
                    "vkCmdBindDescriptorSets",
                    _
                ),
                cmd_push_constants: load_vk_fn!(get_proc, instance, "vkCmdPushConstants", _),
                cmd_dispatch: load_vk_fn!(get_proc, instance, "vkCmdDispatch", _),
                cmd_copy_buffer: load_vk_fn!(get_proc, instance, "vkCmdCopyBuffer", _),
                cmd_pipeline_barrier: load_vk_fn!(get_proc, instance, "vkCmdPipelineBarrier", _),
                create_fence: load_vk_fn!(get_proc, instance, "vkCreateFence", _),
                destroy_fence: load_vk_fn!(get_proc, instance, "vkDestroyFence", _),
                wait_for_fences: load_vk_fn!(get_proc, instance, "vkWaitForFences", _),
                reset_fences: load_vk_fn!(get_proc, instance, "vkResetFences", _),
                queue_submit: load_vk_fn!(get_proc, instance, "vkQueueSubmit", _),
            };

            let mut pd_count = 0u32;
            let res = (fns.enumerate_physical_devices)(instance, &mut pd_count, ptr::null_mut());
            if res != VK_SUCCESS || pd_count == 0 {
                (fns.destroy_instance)(instance, ptr::null());
                return Err("Nenhum dispositivo físico Vulkan encontrado.".to_string());
            }
            let mut pds = vec![ptr::null_mut(); pd_count as usize];
            let res = (fns.enumerate_physical_devices)(instance, &mut pd_count, pds.as_mut_ptr());
            if res != VK_SUCCESS {
                (fns.destroy_instance)(instance, ptr::null());
                return Err(format!("vkEnumeratePhysicalDevices falhou ({}).", res));
            }

            // Seleciona a melhor GPU com fila Compute (prioriza DISCRETE_GPU como a RX 7600)
            let mut chosen: Option<(
                VkPhysicalDevice,
                u32,
                VulkanAdapterInfo,
                VkPhysicalDeviceMemoryProperties,
                i32,
            )> = None;

            for pd in pds {
                let mut raw_props = RawPhysicalDeviceProperties { bytes: [0u8; 1024] };
                (fns.get_physical_device_properties)(pd, &mut raw_props);
                let b = &raw_props.bytes;
                let api_ver = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let vendor_id = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
                let device_id = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
                let dev_type = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
                let max_sb_range =
                    u32::from_le_bytes([b[324], b[325], b[326], b[327]]) as u64;
                let max_pc_size = u32::from_le_bytes([b[328], b[329], b[330], b[331]]);

                if api_ver < VK_API_VERSION_1_1 || max_pc_size < PUSH_CONSTANT_BYTES {
                    continue;
                }

                let name_bytes = &b[20..276];
                let nul_pos = name_bytes
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(name_bytes.len());
                let device_name = String::from_utf8_lossy(&name_bytes[..nul_pos]).to_string();

                let mut qf_count = 0u32;
                (fns.get_physical_device_queue_family_properties)(pd, &mut qf_count, ptr::null_mut());
                if qf_count == 0 {
                    continue;
                }
                let mut qfs = vec![
                    VkQueueFamilyProperties {
                        queue_flags: 0,
                        queue_count: 0,
                        timestamp_valid_bits: 0,
                        min_image_transfer_granularity: [0; 3],
                    };
                    qf_count as usize
                ];
                (fns.get_physical_device_queue_family_properties)(
                    pd,
                    &mut qf_count,
                    qfs.as_mut_ptr(),
                );
                let Some(qf_idx) = qfs
                    .iter()
                    .position(|q| (q.queue_flags & VK_QUEUE_COMPUTE_BIT) != 0 && q.queue_count > 0)
                    .map(|i| i as u32)
                else {
                    continue;
                };

                let mut mem_props = std::mem::zeroed::<VkPhysicalDeviceMemoryProperties>();
                (fns.get_physical_device_memory_properties)(pd, &mut mem_props);
                let mut vram_bytes = 0u64;
                for h_idx in 0..(mem_props.memory_heap_count as usize).min(16) {
                    if (mem_props.memory_heaps[h_idx].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 {
                        vram_bytes = vram_bytes.max(mem_props.memory_heaps[h_idx].size);
                    }
                }

                let score = match dev_type {
                    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 100,
                    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 50,
                    _ => 10,
                };
                let info = VulkanAdapterInfo {
                    device_name,
                    vendor_id,
                    device_id,
                    is_discrete_gpu: dev_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                    api_version_str: format!(
                        "{}.{}.{}",
                        (api_ver >> 22) & 0x7F,
                        (api_ver >> 12) & 0x3FF,
                        api_ver & 0xFFF
                    ),
                    device_local_vram_mb: vram_bytes / (1024 * 1024),
                    max_storage_buffer_range: max_sb_range,
                };

                if chosen.as_ref().map_or(true, |c| score > c.4) {
                    chosen = Some((pd, qf_idx, info, mem_props, score));
                }
            }

            let Some((phys_dev, queue_family, info, mem_props, _)) = chosen else {
                (fns.destroy_instance)(instance, ptr::null());
                return Err("Nenhuma GPU compatível com Vulkan 1.1 Compute encontrada.".to_string());
            };

            let priority = 1.0f32;
            let q_ci = VkDeviceQueueCreateInfo {
                s_type: VK_STYPE_DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                queue_family_index: queue_family,
                queue_count: 1,
                p_queue_priorities: &priority,
            };
            let dev_ci = VkDeviceCreateInfo {
                s_type: VK_STYPE_DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                queue_create_info_count: 1,
                p_queue_create_infos: &q_ci,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
                p_enabled_features: ptr::null(),
            };

            let mut device: VkDevice = ptr::null_mut();
            let res = (fns.create_device)(phys_dev, &dev_ci, ptr::null(), &mut device);
            if res != VK_SUCCESS || device.is_null() {
                (fns.destroy_instance)(instance, ptr::null());
                return Err(format!("vkCreateDevice falhou ({}).", res));
            }

            let mut queue: VkQueue = ptr::null_mut();
            (fns.get_device_queue)(device, queue_family, 0, &mut queue);

            // Cria DescriptorSetLayout (binding 0 = Arena f32[], binding 1 = IO u32[])
            let bindings = [
                VkDescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
                VkDescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
            ];
            let dsl_ci = VkDescriptorSetLayoutCreateInfo {
                s_type: VK_STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
            };
            let mut desc_set_layout: VkDescriptorSetLayout = 0;
            let res = (fns.create_descriptor_set_layout)(
                device,
                &dsl_ci,
                ptr::null(),
                &mut desc_set_layout,
            );
            if res != VK_SUCCESS {
                (fns.destroy_device)(device, ptr::null());
                (fns.destroy_instance)(instance, ptr::null());
                return Err(format!("vkCreateDescriptorSetLayout falhou ({}).", res));
            }

            let pc_range = VkPushConstantRange {
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                offset: 0,
                size: PUSH_CONSTANT_BYTES,
            };
            let pl_ci = VkPipelineLayoutCreateInfo {
                s_type: VK_STYPE_PIPELINE_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                set_layout_count: 1,
                p_set_layouts: &desc_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &pc_range,
            };
            let mut pipeline_layout: VkPipelineLayout = 0;
            let res =
                (fns.create_pipeline_layout)(device, &pl_ci, ptr::null(), &mut pipeline_layout);
            if res != VK_SUCCESS {
                (fns.destroy_descriptor_set_layout)(device, desc_set_layout, ptr::null());
                (fns.destroy_device)(device, ptr::null());
                (fns.destroy_instance)(instance, ptr::null());
                return Err(format!("vkCreatePipelineLayout falhou ({}).", res));
            }

            let spv = NeuralSpirvKernels::build_all();
            let compile = |label: &str, words: &[u32]| -> Result<VkPipeline, String> {
                Self::compile_compute_pipeline(&fns, device, pipeline_layout, label, words)
            };

            let pipelines_res = (|| -> Result<CompiledPipelines, String> {
                Ok(CompiledPipelines {
                    fill_f32: compile("fill_f32", &spv.fill_f32)?,
                    vec_add_inplace: compile("vec_add_inplace", &spv.vec_add_inplace)?,
                    embed_fwd: compile("embed_fwd", &spv.embed_fwd)?,
                    embed_bwd: compile("embed_bwd", &spv.embed_bwd)?,
                    rmsnorm_fwd: compile("rmsnorm_fwd", &spv.rmsnorm_fwd)?,
                    rmsnorm_bwd_dx: compile("rmsnorm_bwd_dx", &spv.rmsnorm_bwd_dx)?,
                    rmsnorm_bwd_dgamma: compile("rmsnorm_bwd_dgamma", &spv.rmsnorm_bwd_dgamma)?,
                    matmul_nn: compile("matmul_nn_tiled", &spv.matmul_nn_tiled)?,
                    matmul_tn: compile("matmul_tn_tiled", &spv.matmul_tn_tiled)?,
                    matmul_nt: compile("matmul_nt_tiled", &spv.matmul_nt_tiled)?,
                    causal_mha_fwd: compile("causal_mha_fwd", &spv.causal_mha_fwd)?,
                    causal_mha_bwd_q_ds: compile("causal_mha_bwd_q_ds", &spv.causal_mha_bwd_q_ds)?,
                    causal_mha_bwd_kv: compile("causal_mha_bwd_kv", &spv.causal_mha_bwd_kv)?,
                    bias_gelu_fwd: compile("bias_gelu_fwd", &spv.bias_gelu_fwd)?,
                    gelu_bwd: compile("gelu_bwd", &spv.gelu_bwd)?,
                    bias_res_add_fwd: compile("bias_res_add_fwd", &spv.bias_res_add_fwd)?,
                    bias_grad_sum: compile("bias_grad_sum", &spv.bias_grad_sum)?,
                    cross_entropy_fwd_bwd: compile(
                        "cross_entropy_fwd_bwd",
                        &spv.cross_entropy_fwd_bwd,
                    )?,
                    reduce_sum_f32_to_io: compile(
                        "reduce_sum_f32_to_io",
                        &spv.reduce_sum_f32_to_io,
                    )?,
                    scale_and_grad_sq: compile("scale_and_grad_sq", &spv.scale_and_grad_sq)?,
                    adamw_step: compile("adamw_step", &spv.adamw_step)?,
                })
            })();

            let pipelines = match pipelines_res {
                Ok(p) => p,
                Err(e) => {
                    (fns.destroy_pipeline_layout)(device, pipeline_layout, ptr::null());
                    (fns.destroy_descriptor_set_layout)(device, desc_set_layout, ptr::null());
                    (fns.destroy_device)(device, ptr::null());
                    (fns.destroy_instance)(instance, ptr::null());
                    return Err(e);
                }
            };

            Ok(Self {
                info,
                instance,
                device,
                queue_family,
                queue_lock: Mutex::new(queue),
                mem_props,
                desc_set_layout,
                pipeline_layout,
                pipelines,
                fns,
            })
        }
    }

    unsafe fn compile_compute_pipeline(
        fns: &VkFns,
        device: VkDevice,
        layout: VkPipelineLayout,
        label: &str,
        words: &[u32],
    ) -> Result<VkPipeline, String> {
        let sm_ci = VkShaderModuleCreateInfo {
            s_type: VK_STYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: words.len() * 4,
            p_code: words.as_ptr(),
        };
        let mut module: VkShaderModule = 0;
        let res = (fns.create_shader_module)(device, &sm_ci, ptr::null(), &mut module);
        if res != VK_SUCCESS || module == 0 {
            return Err(format!(
                "vkCreateShaderModule falhou para '{}' ({}).",
                label, res
            ));
        }

        let main_cstr = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let cp_ci = VkComputePipelineCreateInfo {
            s_type: VK_STYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: VkPipelineShaderStageCreateInfo {
                s_type: VK_STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module,
                p_name: main_cstr.as_ptr(),
                p_specialization_info: ptr::null(),
            },
            layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut pipeline: VkPipeline = 0;
        let res = (fns.create_compute_pipelines)(device, 0, 1, &cp_ci, ptr::null(), &mut pipeline);
        (fns.destroy_shader_module)(device, module, ptr::null());
        if res != VK_SUCCESS || pipeline == 0 {
            return Err(format!(
                "vkCreateComputePipelines falhou para '{}' ({}).",
                label, res
            ));
        }
        Ok(pipeline)
    }

    fn find_memory_type(
        &self,
        type_bits: u32,
        required: u32,
        preferred_not: u32,
    ) -> Option<u32> {
        let count = (self.mem_props.memory_type_count as usize).min(32);
        // 1ª tentativa: satisfaz `required` e evita `preferred_not` (ex.: VRAM dedicada pura)
        if preferred_not != 0 {
            for i in 0..count {
                let ok_bit = (type_bits & (1 << i)) != 0;
                let flags = self.mem_props.memory_types[i].property_flags;
                if ok_bit && (flags & required) == required && (flags & preferred_not) == 0 {
                    return Some(i as u32);
                }
            }
        }
        // 2ª tentativa: qualquer tipo compatível com `required`
        for i in 0..count {
            let ok_bit = (type_bits & (1 << i)) != 0;
            let flags = self.mem_props.memory_types[i].property_flags;
            if ok_bit && (flags & required) == required {
                return Some(i as u32);
            }
        }
        None
    }

    unsafe fn alloc_buffer(
        &self,
        size_bytes: u64,
        usage: u32,
        req_mem: u32,
        avoid_mem: u32,
    ) -> Result<(VkBuffer, VkDeviceMemory), String> {
        let qf = self.queue_family;
        let b_ci = VkBufferCreateInfo {
            s_type: VK_STYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size: size_bytes.max(16),
            usage,
            sharing_mode: 0,
            queue_family_index_count: 1,
            p_queue_family_indices: &qf,
        };
        let mut buf: VkBuffer = 0;
        let res = (self.fns.create_buffer)(self.device, &b_ci, ptr::null(), &mut buf);
        if res != VK_SUCCESS || buf == 0 {
            return Err(format!("vkCreateBuffer ({} bytes) falhou ({}).", size_bytes, res));
        }

        let mut req = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        (self.fns.get_buffer_memory_requirements)(self.device, buf, &mut req);

        let Some(mem_type) = self.find_memory_type(req.memory_type_bits, req_mem, avoid_mem) else {
            (self.fns.destroy_buffer)(self.device, buf, ptr::null());
            return Err("Tipo de memória Vulkan compatível não encontrado.".to_string());
        };

        let a_ci = VkMemoryAllocateInfo {
            s_type: VK_STYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: req.size,
            memory_type_index: mem_type,
        };
        let mut mem: VkDeviceMemory = 0;
        let res = (self.fns.allocate_memory)(self.device, &a_ci, ptr::null(), &mut mem);
        if res != VK_SUCCESS || mem == 0 {
            (self.fns.destroy_buffer)(self.device, buf, ptr::null());
            return Err(format!(
                "vkAllocateMemory ({:.2} MB) falhou ({}).",
                req.size as f64 / (1024.0 * 1024.0),
                res
            ));
        }

        let res = (self.fns.bind_buffer_memory)(self.device, buf, mem, 0);
        if res != VK_SUCCESS {
            (self.fns.free_memory)(self.device, mem, ptr::null());
            (self.fns.destroy_buffer)(self.device, buf, ptr::null());
            return Err(format!("vkBindBufferMemory falhou ({}).", res));
        }

        Ok((buf, mem))
    }

    /// Executa benchmark puro de GEMM FP32 residente na VRAM da GPU (`size × size`),
    /// medindo GFLOP/s reais em `DEVICE_LOCAL` sem poluição de transferência PCIe.
    pub fn benchmark_vram_matmul(&self, size: usize) -> Result<(f64, f64), String> {
        let n = size.max(64) as u32;
        let elems = (n as usize) * (n as usize);
        let arena_bytes = (elems * 3 * 4) as u64;

        unsafe {
            let (arena_buf, arena_mem) = self.alloc_buffer(
                arena_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            )?;
            let (io_buf, io_mem) = self.alloc_buffer(
                64,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                0,
            )?;

            let res = (|| -> Result<(f64, f64), String> {
                let pool_size = VkDescriptorPoolSize {
                    ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptor_count: 2,
                };
                let dp_ci = VkDescriptorPoolCreateInfo {
                    s_type: VK_STYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: 0,
                    max_sets: 1,
                    pool_size_count: 1,
                    p_pool_sizes: &pool_size,
                };
                let mut dpool: VkDescriptorPool = 0;
                if (self.fns.create_descriptor_pool)(self.device, &dp_ci, ptr::null(), &mut dpool)
                    != VK_SUCCESS
                {
                    return Err("Falha ao criar DescriptorPool de benchmark.".to_string());
                }

                let ds_ai = VkDescriptorSetAllocateInfo {
                    s_type: VK_STYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    p_next: ptr::null(),
                    descriptor_pool: dpool,
                    descriptor_set_count: 1,
                    p_set_layouts: &self.desc_set_layout,
                };
                let mut dset: VkDescriptorSet = 0;
                let _ = (self.fns.allocate_descriptor_sets)(self.device, &ds_ai, &mut dset);
                let b_infos = [
                    VkDescriptorBufferInfo {
                        buffer: arena_buf,
                        offset: 0,
                        range: VK_WHOLE_SIZE,
                    },
                    VkDescriptorBufferInfo {
                        buffer: io_buf,
                        offset: 0,
                        range: VK_WHOLE_SIZE,
                    },
                ];
                let writes = [
                    VkWriteDescriptorSet {
                        s_type: VK_STYPE_WRITE_DESCRIPTOR_SET,
                        p_next: ptr::null(),
                        dst_set: dset,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        p_image_info: ptr::null(),
                        p_buffer_info: &b_infos[0],
                        p_texel_buffer_view: ptr::null(),
                    },
                    VkWriteDescriptorSet {
                        s_type: VK_STYPE_WRITE_DESCRIPTOR_SET,
                        p_next: ptr::null(),
                        dst_set: dset,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        p_image_info: ptr::null(),
                        p_buffer_info: &b_infos[1],
                        p_texel_buffer_view: ptr::null(),
                    },
                ];
                (self.fns.update_descriptor_sets)(
                    self.device,
                    writes.len() as u32,
                    writes.as_ptr(),
                    0,
                    ptr::null(),
                );

                let cp_ci = VkCommandPoolCreateInfo {
                    s_type: VK_STYPE_COMMAND_POOL_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queue_family_index: self.queue_family,
                };
                let mut cpool: VkCommandPool = 0;
                let _ =
                    (self.fns.create_command_pool)(self.device, &cp_ci, ptr::null(), &mut cpool);
                let cb_ai = VkCommandBufferAllocateInfo {
                    s_type: VK_STYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: ptr::null(),
                    command_pool: cpool,
                    level: 0,
                    command_buffer_count: 1,
                };
                let mut cmd: VkCommandBuffer = ptr::null_mut();
                let _ = (self.fns.allocate_command_buffers)(self.device, &cb_ai, &mut cmd);

                let f_ci = VkFenceCreateInfo {
                    s_type: VK_STYPE_FENCE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: 0,
                };
                let mut fence: VkFence = 0;
                let _ = (self.fns.create_fence)(self.device, &f_ci, ptr::null(), &mut fence);

                let wg = (n + 15) / 16;
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = 0;
                pc[1] = elems as u32;
                pc[2] = (elems * 2) as u32;
                pc[3] = n;
                pc[4] = n;
                pc[5] = n;

                let run_iters = |iters: usize| -> Result<f64, String> {
                    let begin = VkCommandBufferBeginInfo {
                        s_type: VK_STYPE_COMMAND_BUFFER_BEGIN_INFO,
                        p_next: ptr::null(),
                        flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                        p_inheritance_info: ptr::null(),
                    };
                    let _ = (self.fns.reset_command_buffer)(cmd, 0);
                    let _ = (self.fns.begin_command_buffer)(cmd, &begin);
                    (self.fns.cmd_bind_descriptor_sets)(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        self.pipeline_layout,
                        0,
                        1,
                        &dset,
                        0,
                        ptr::null(),
                    );
                    (self.fns.cmd_bind_pipeline)(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        self.pipelines.matmul_nn,
                    );
                    (self.fns.cmd_push_constants)(
                        cmd,
                        self.pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        PUSH_CONSTANT_BYTES,
                        pc.as_ptr() as *const c_void,
                    );
                    let mb = VkMemoryBarrier {
                        s_type: VK_STYPE_MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                        dst_access_mask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    };
                    for k_it in 0..iters {
                        (self.fns.cmd_dispatch)(cmd, wg, wg, 1);
                        if k_it + 1 < iters {
                            (self.fns.cmd_pipeline_barrier)(
                                cmd,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0,
                                1,
                                &mb,
                                0,
                                ptr::null(),
                                0,
                                ptr::null(),
                            );
                        }
                    }
                    let _ = (self.fns.end_command_buffer)(cmd);

                    let q = self
                        .queue_lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let _ = (self.fns.reset_fences)(self.device, 1, &fence);
                    let sub = VkSubmitInfo {
                        s_type: VK_STYPE_SUBMIT_INFO,
                        p_next: ptr::null(),
                        wait_semaphore_count: 0,
                        p_wait_semaphores: ptr::null(),
                        p_wait_dst_stage_mask: ptr::null(),
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        signal_semaphore_count: 0,
                        p_signal_semaphores: ptr::null(),
                    };
                    let t0 = std::time::Instant::now();
                    let sres = (self.fns.queue_submit)(*q, 1, &sub, fence);
                    if sres != VK_SUCCESS {
                        return Err(format!("vkQueueSubmit falhou ({}).", sres));
                    }
                    let wres = (self.fns.wait_for_fences)(self.device, 1, &fence, 1, u64::MAX);
                    let secs = t0.elapsed().as_secs_f64();
                    drop(q);
                    if wres != VK_SUCCESS {
                        return Err(format!("vkWaitForFences falhou ({}).", wres));
                    }
                    Ok(secs)
                };

                // Inicializa buffers com 0.25 e faz warmup
                {
                    let mut fill_pc = [0u32; PUSH_CONSTANT_WORDS];
                    fill_pc[0] = 0;
                    fill_pc[1] = (elems * 3) as u32;
                    fill_pc[2] = 0.25f32.to_bits();
                    let begin = VkCommandBufferBeginInfo {
                        s_type: VK_STYPE_COMMAND_BUFFER_BEGIN_INFO,
                        p_next: ptr::null(),
                        flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                        p_inheritance_info: ptr::null(),
                    };
                    let _ = (self.fns.begin_command_buffer)(cmd, &begin);
                    (self.fns.cmd_bind_descriptor_sets)(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        self.pipeline_layout,
                        0,
                        1,
                        &dset,
                        0,
                        ptr::null(),
                    );
                    (self.fns.cmd_bind_pipeline)(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        self.pipelines.fill_f32,
                    );
                    (self.fns.cmd_push_constants)(
                        cmd,
                        self.pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        PUSH_CONSTANT_BYTES,
                        fill_pc.as_ptr() as *const c_void,
                    );
                    (self.fns.cmd_dispatch)(cmd, ((elems * 3 + 255) / 256) as u32, 1, 1);
                    let _ = (self.fns.end_command_buffer)(cmd);
                    let q = self.queue_lock.lock().unwrap_or_else(|p| p.into_inner());
                    let _ = (self.fns.reset_fences)(self.device, 1, &fence);
                    let sub = VkSubmitInfo {
                        s_type: VK_STYPE_SUBMIT_INFO,
                        p_next: ptr::null(),
                        wait_semaphore_count: 0,
                        p_wait_semaphores: ptr::null(),
                        p_wait_dst_stage_mask: ptr::null(),
                        command_buffer_count: 1,
                        p_command_buffers: &cmd,
                        signal_semaphore_count: 0,
                        p_signal_semaphores: ptr::null(),
                    };
                    let _ = (self.fns.queue_submit)(*q, 1, &sub, fence);
                    let _ = (self.fns.wait_for_fences)(self.device, 1, &fence, 1, u64::MAX);
                }

                let _ = run_iters(2)?;
                let iters = 16usize;
                let secs = run_iters(iters)?;
                let flops = 2.0 * (iters as f64) * (n as f64).powi(3);
                let gflops = (flops / secs.max(1e-9)) / 1e9;
                let avg_ms = (secs * 1000.0) / (iters as f64);

                (self.fns.destroy_fence)(self.device, fence, ptr::null());
                (self.fns.destroy_command_pool)(self.device, cpool, ptr::null());
                (self.fns.destroy_descriptor_pool)(self.device, dpool, ptr::null());
                Ok((avg_ms, gflops))
            })();

            (self.fns.free_memory)(self.device, io_mem, ptr::null());
            (self.fns.destroy_buffer)(self.device, io_buf, ptr::null());
            (self.fns.free_memory)(self.device, arena_mem, ptr::null());
            (self.fns.destroy_buffer)(self.device, arena_buf, ptr::null());
            res
        }
    }
}

// ===========================================================================
// Sessão de Treinamento / Avaliação Residente em VRAM (`VulkanTrainingSession`)
// ===========================================================================

#[derive(Clone, Copy)]
struct LayerParamOffsets {
    ln1_gamma: u32,
    wq: u32,
    wk: u32,
    wv: u32,
    wo: u32,
    ln2_gamma: u32,
    w1: u32,
    b1: u32,
    w2: u32,
    b2: u32,
}

#[derive(Clone, Copy)]
struct LayerActOffsets {
    x_in: u32,
    ln1: u32,
    inv_rms1: u32,
    q: u32,
    k: u32,
    v: u32,
    attn_probs: u32,
    attn_ctx: u32,
    x_mid: u32,
    ln2: u32,
    inv_rms2: u32,
    h_pre: u32,
    h_act: u32,
}

#[derive(Clone, Copy)]
struct AdamGroup {
    rel_off: u32,
    len: u32,
    apply_decay: bool,
}

/// Sessão persistente na VRAM da RX 7600 mantendo todos os pesos, gradientes, momentos `m`/`v`
/// do AdamW e tensores de ativação em memória dedicada `DEVICE_LOCAL`.
pub struct VulkanTrainingSession {
    dev: Arc<VulkanDevice>,
    config: TransformerConfig,
    total_params: u32,
    arena_total_floats: usize,
    vram_allocated_bytes: u64,

    // Offsets base em unidades de `f32` dentro de `arena_buf`
    off_weights: u32,
    off_grads: u32,
    off_m: u32,
    off_v: u32,

    // Offsets relativos de parâmetros (aplicáveis a `off_weights`, `off_grads`, `off_m`, `off_v`)
    rel_tok_emb: u32,
    rel_pos_emb: u32,
    rel_layers: Vec<LayerParamOffsets>,
    rel_ln_f_gamma: u32,
    rel_lm_head: u32,
    adam_groups: Vec<AdamGroup>,

    // Offsets de Workspace em `arena_buf`
    act_layers: Vec<LayerActOffsets>,
    ws_x: u32,
    ws_attn_out: u32,
    ws_mlp_out: u32,
    ws_ln_f: u32,
    ws_inv_rms_f: u32,
    ws_logits: u32,
    ws_d_logits: u32,
    ws_d_ln_f: u32,
    ws_d_x: u32,
    ws_d_x_mid: u32,
    ws_d_x_in: u32,
    ws_d_rms_tmp: u32,
    ws_d_h_act: u32,
    ws_d_h_pre: u32,
    ws_d_ln2: u32,
    ws_d_attn_ctx: u32,
    ws_d_scores: u32,
    ws_d_q: u32,
    ws_d_k: u32,
    ws_d_v: u32,
    ws_d_ln1: u32,
    ws_row_losses: u32,
    ws_grad_partial: u32,

    // Layout em palavras `u32` dentro de `io_buf`
    io_in_off: u32,
    io_tgt_off: u32,
    io_out_loss: u32,
    io_out_grad_sq: u32,

    // Recursos Vulkan da sessão
    arena_buf: VkBuffer,
    arena_mem: VkDeviceMemory,
    io_buf: VkBuffer,
    io_mem: VkDeviceMemory,
    io_mapped: *mut u32,
    staging_buf: VkBuffer,
    staging_mem: VkDeviceMemory,
    staging_mapped: *mut f32,
    staging_floats: usize,
    desc_pool: VkDescriptorPool,
    desc_set: VkDescriptorSet,
    cmd_pool: VkCommandPool,
    cmd_buf: VkCommandBuffer,
    fence: VkFence,
}

unsafe impl Send for VulkanTrainingSession {}

impl Drop for VulkanTrainingSession {
    fn drop(&mut self) {
        unsafe {
            let f = &self.dev.fns;
            let d = self.dev.device;
            if self.fence != 0 {
                (f.destroy_fence)(d, self.fence, ptr::null());
            }
            if self.cmd_pool != 0 {
                (f.destroy_command_pool)(d, self.cmd_pool, ptr::null());
            }
            if self.desc_pool != 0 {
                (f.destroy_descriptor_pool)(d, self.desc_pool, ptr::null());
            }
            if !self.staging_mapped.is_null() {
                (f.unmap_memory)(d, self.staging_mem);
            }
            if self.staging_buf != 0 {
                (f.destroy_buffer)(d, self.staging_buf, ptr::null());
            }
            if self.staging_mem != 0 {
                (f.free_memory)(d, self.staging_mem, ptr::null());
            }
            if !self.io_mapped.is_null() {
                (f.unmap_memory)(d, self.io_mem);
            }
            if self.io_buf != 0 {
                (f.destroy_buffer)(d, self.io_buf, ptr::null());
            }
            if self.io_mem != 0 {
                (f.free_memory)(d, self.io_mem, ptr::null());
            }
            if self.arena_buf != 0 {
                (f.destroy_buffer)(d, self.arena_buf, ptr::null());
            }
            if self.arena_mem != 0 {
                (f.free_memory)(d, self.arena_mem, ptr::null());
            }
        }
    }
}

impl VulkanTrainingSession {
    /// Inicializa a sessão na GPU e faz o upload inicial de `model` e `optimizer` para a VRAM.
    pub fn new(
        dev: Arc<VulkanDevice>,
        model: &ServerMindTransformer,
        optimizer: &AdamWState,
    ) -> Result<Self, String> {
        model.config.validate()?;
        let cfg = model.config.clone();
        let total_params_usize = cfg
            .checked_total_params()
            .ok_or_else(|| "Overflow no total de parâmetros ao alocar sessão GPU.".to_string())?;
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let v = cfg.vocab_size;
        let max_t = cfg.max_seq_len;
        let h = cfg.n_heads;

        // 1. Mapeia offsets relativos idênticos à ordem de `visit_param_slices` / `AdamWState::step`
        let mut rel_cur = 0usize;
        let mut alloc_rel = |len: usize| -> Result<u32, String> {
            let start = u32::try_from(rel_cur)
                .map_err(|_| "Offset de parâmetro excede limite u32.".to_string())?;
            rel_cur = rel_cur
                .checked_add(len)
                .ok_or_else(|| "Overflow em offset relativo.".to_string())?;
            Ok(start)
        };

        let mut raw_groups: Vec<AdamGroup> = Vec::new();
        let mut push_group = |rel_off: u32, len: usize, apply_decay: bool| {
            let len_u = len as u32;
            if let Some(last) = raw_groups.last_mut() {
                if last.apply_decay == apply_decay && last.rel_off + last.len == rel_off {
                    last.len += len_u;
                    return;
                }
            }
            raw_groups.push(AdamGroup {
                rel_off,
                len: len_u,
                apply_decay,
            });
        };

        let rel_tok_emb = alloc_rel(v * d)?;
        push_group(rel_tok_emb, v * d, false);
        let rel_pos_emb = alloc_rel(max_t * d)?;
        push_group(rel_pos_emb, max_t * d, false);

        let mut rel_layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let ln1_gamma = alloc_rel(d)?;
            push_group(ln1_gamma, d, false);
            let wq = alloc_rel(d * d)?;
            push_group(wq, d * d, true);
            let wk = alloc_rel(d * d)?;
            push_group(wk, d * d, true);
            let wv = alloc_rel(d * d)?;
            push_group(wv, d * d, true);
            let wo = alloc_rel(d * d)?;
            push_group(wo, d * d, true);
            let ln2_gamma = alloc_rel(d)?;
            push_group(ln2_gamma, d, false);
            let w1 = alloc_rel(d * ff)?;
            push_group(w1, d * ff, true);
            let b1 = alloc_rel(ff)?;
            push_group(b1, ff, false);
            let w2 = alloc_rel(ff * d)?;
            push_group(w2, ff * d, true);
            let b2 = alloc_rel(d)?;
            push_group(b2, d, false);
            rel_layers.push(LayerParamOffsets {
                ln1_gamma,
                wq,
                wk,
                wv,
                wo,
                ln2_gamma,
                w1,
                b1,
                w2,
                b2,
            });
        }
        let rel_ln_f_gamma = alloc_rel(d)?;
        push_group(rel_ln_f_gamma, d, false);
        let rel_lm_head = alloc_rel(d * v)?;
        push_group(rel_lm_head, d * v, true);
        assert_eq!(rel_cur, total_params_usize);

        // 2. Mapeia offsets globais dentro da Arena VRAM
        let mut arena_cur = 0usize;
        let mut alloc_arena = |len: usize| -> Result<u32, String> {
            let start = u32::try_from(arena_cur)
                .map_err(|_| "Arena VRAM excede indexação u32.".to_string())?;
            arena_cur = arena_cur
                .checked_add(len)
                .ok_or_else(|| "Overflow ao dimensionar Arena VRAM.".to_string())?;
            Ok(start)
        };

        let off_weights = alloc_arena(total_params_usize)?;
        let off_grads = alloc_arena(total_params_usize)?;
        let off_m = alloc_arena(total_params_usize)?;
        let off_v = alloc_arena(total_params_usize)?;

        let td = max_t * d;
        let tff = max_t * ff;
        let tv = max_t * v;
        let htt = h * max_t * max_t;

        let mut act_layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            act_layers.push(LayerActOffsets {
                x_in: alloc_arena(td)?,
                ln1: alloc_arena(td)?,
                inv_rms1: alloc_arena(max_t)?,
                q: alloc_arena(td)?,
                k: alloc_arena(td)?,
                v: alloc_arena(td)?,
                attn_probs: alloc_arena(htt)?,
                attn_ctx: alloc_arena(td)?,
                x_mid: alloc_arena(td)?,
                ln2: alloc_arena(td)?,
                inv_rms2: alloc_arena(max_t)?,
                h_pre: alloc_arena(tff)?,
                h_act: alloc_arena(tff)?,
            });
        }

        let ws_x = alloc_arena(td)?;
        let ws_attn_out = alloc_arena(td)?;
        let ws_mlp_out = alloc_arena(td)?;
        let ws_ln_f = alloc_arena(td)?;
        let ws_inv_rms_f = alloc_arena(max_t)?;
        let ws_logits = alloc_arena(tv)?;
        let ws_d_logits = alloc_arena(tv)?;
        let ws_d_ln_f = alloc_arena(td)?;
        let ws_d_x = alloc_arena(td)?;
        let ws_d_x_mid = alloc_arena(td)?;
        let ws_d_x_in = alloc_arena(td)?;
        let ws_d_rms_tmp = alloc_arena(td)?;
        let ws_d_h_act = alloc_arena(tff)?;
        let ws_d_h_pre = alloc_arena(tff)?;
        let ws_d_ln2 = alloc_arena(td)?;
        let ws_d_attn_ctx = alloc_arena(td)?;
        let ws_d_scores = alloc_arena(htt)?;
        let ws_d_q = alloc_arena(td)?;
        let ws_d_k = alloc_arena(td)?;
        let ws_d_v = alloc_arena(td)?;
        let ws_d_ln1 = alloc_arena(td)?;
        let ws_row_losses = alloc_arena(max_t)?;
        let num_grad_wgs = (total_params_usize + 255) / 256;
        let ws_grad_partial = alloc_arena(num_grad_wgs.max(1))?;

        let arena_bytes = (arena_cur as u64) * 4;
        if arena_bytes > dev.info.max_storage_buffer_range {
            return Err(format!(
                "Arena VRAM requerida ({:.1} MB) excede maxStorageBufferRange ({:.1} MB).",
                arena_bytes as f64 / 1_048_576.0,
                dev.info.max_storage_buffer_range as f64 / 1_048_576.0
            ));
        }

        let io_words = max_t * 2 + 16;
        let io_bytes = (io_words * 4) as u64;
        // Staging dimensionado para `3 * total_params` (pesos + AdamW `m` + `v` em uma única cópia)
        let staging_floats = total_params_usize * 3;
        let staging_bytes = (staging_floats * 4) as u64;

        unsafe {
            let (arena_buf, arena_mem) = dev.alloc_buffer(
                arena_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            )?;

            let (io_buf, io_mem) = match dev.alloc_buffer(
                io_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                0,
            ) {
                Ok(pair) => pair,
                Err(e) => {
                    (dev.fns.destroy_buffer)(dev.device, arena_buf, ptr::null());
                    (dev.fns.free_memory)(dev.device, arena_mem, ptr::null());
                    return Err(e);
                }
            };

            let (staging_buf, staging_mem) = match dev.alloc_buffer(
                staging_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                0,
            ) {
                Ok(pair) => pair,
                Err(e) => {
                    (dev.fns.destroy_buffer)(dev.device, io_buf, ptr::null());
                    (dev.fns.free_memory)(dev.device, io_mem, ptr::null());
                    (dev.fns.destroy_buffer)(dev.device, arena_buf, ptr::null());
                    (dev.fns.free_memory)(dev.device, arena_mem, ptr::null());
                    return Err(e);
                }
            };

            let mut sess = Self {
                dev: Arc::clone(&dev),
                config: cfg,
                total_params: total_params_usize as u32,
                arena_total_floats: arena_cur,
                vram_allocated_bytes: arena_bytes,
                off_weights,
                off_grads,
                off_m,
                off_v,
                rel_tok_emb,
                rel_pos_emb,
                rel_layers,
                rel_ln_f_gamma,
                rel_lm_head,
                adam_groups: raw_groups,
                act_layers,
                ws_x,
                ws_attn_out,
                ws_mlp_out,
                ws_ln_f,
                ws_inv_rms_f,
                ws_logits,
                ws_d_logits,
                ws_d_ln_f,
                ws_d_x,
                ws_d_x_mid,
                ws_d_x_in,
                ws_d_rms_tmp,
                ws_d_h_act,
                ws_d_h_pre,
                ws_d_ln2,
                ws_d_attn_ctx,
                ws_d_scores,
                ws_d_q,
                ws_d_k,
                ws_d_v,
                ws_d_ln1,
                ws_row_losses,
                ws_grad_partial,
                io_in_off: 0,
                io_tgt_off: max_t as u32,
                io_out_loss: (max_t * 2) as u32,
                io_out_grad_sq: (max_t * 2 + 1) as u32,
                arena_buf,
                arena_mem,
                io_buf,
                io_mem,
                io_mapped: ptr::null_mut(),
                staging_buf,
                staging_mem,
                staging_mapped: ptr::null_mut(),
                staging_floats,
                desc_pool: 0,
                desc_set: 0,
                cmd_pool: 0,
                cmd_buf: ptr::null_mut(),
                fence: 0,
            };

            let mut raw_io: *mut c_void = ptr::null_mut();
            if (dev.fns.map_memory)(dev.device, io_mem, 0, io_bytes, 0, &mut raw_io) != VK_SUCCESS {
                return Err("Falha em vkMapMemory (io_buf).".to_string());
            }
            sess.io_mapped = raw_io as *mut u32;

            let mut raw_st: *mut c_void = ptr::null_mut();
            if (dev.fns.map_memory)(dev.device, staging_mem, 0, staging_bytes, 0, &mut raw_st)
                != VK_SUCCESS
            {
                return Err("Falha em vkMapMemory (staging_buf).".to_string());
            }
            sess.staging_mapped = raw_st as *mut f32;

            let pool_size = VkDescriptorPoolSize {
                ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 2,
            };
            let dp_ci = VkDescriptorPoolCreateInfo {
                s_type: VK_STYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                max_sets: 1,
                pool_size_count: 1,
                p_pool_sizes: &pool_size,
            };
            if (dev.fns.create_descriptor_pool)(
                dev.device,
                &dp_ci,
                ptr::null(),
                &mut sess.desc_pool,
            ) != VK_SUCCESS
            {
                return Err("Falha em vkCreateDescriptorPool.".to_string());
            }

            let ds_ai = VkDescriptorSetAllocateInfo {
                s_type: VK_STYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: sess.desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: &dev.desc_set_layout,
            };
            if (dev.fns.allocate_descriptor_sets)(dev.device, &ds_ai, &mut sess.desc_set)
                != VK_SUCCESS
            {
                return Err("Falha em vkAllocateDescriptorSets.".to_string());
            }

            let b_infos = [
                VkDescriptorBufferInfo {
                    buffer: sess.arena_buf,
                    offset: 0,
                    range: VK_WHOLE_SIZE,
                },
                VkDescriptorBufferInfo {
                    buffer: sess.io_buf,
                    offset: 0,
                    range: VK_WHOLE_SIZE,
                },
            ];
            let writes = [
                VkWriteDescriptorSet {
                    s_type: VK_STYPE_WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: sess.desc_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: &b_infos[0],
                    p_texel_buffer_view: ptr::null(),
                },
                VkWriteDescriptorSet {
                    s_type: VK_STYPE_WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: sess.desc_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: &b_infos[1],
                    p_texel_buffer_view: ptr::null(),
                },
            ];
            (dev.fns.update_descriptor_sets)(
                dev.device,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                ptr::null(),
            );

            let cp_ci = VkCommandPoolCreateInfo {
                s_type: VK_STYPE_COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queue_family_index: dev.queue_family,
            };
            if (dev.fns.create_command_pool)(dev.device, &cp_ci, ptr::null(), &mut sess.cmd_pool)
                != VK_SUCCESS
            {
                return Err("Falha em vkCreateCommandPool.".to_string());
            }

            let cb_ai = VkCommandBufferAllocateInfo {
                s_type: VK_STYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool: sess.cmd_pool,
                level: 0,
                command_buffer_count: 1,
            };
            if (dev.fns.allocate_command_buffers)(dev.device, &cb_ai, &mut sess.cmd_buf)
                != VK_SUCCESS
            {
                return Err("Falha em vkAllocateCommandBuffers.".to_string());
            }

            let f_ci = VkFenceCreateInfo {
                s_type: VK_STYPE_FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
            };
            if (dev.fns.create_fence)(dev.device, &f_ci, ptr::null(), &mut sess.fence) != VK_SUCCESS
            {
                return Err("Falha em vkCreateFence.".to_string());
            }

            sess.upload_initial_state(model, optimizer)?;
            Ok(sess)
        }
    }

    pub fn vram_allocated_mb(&self) -> f64 {
        self.vram_allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn device_name(&self) -> &str {
        &self.dev.info.device_name
    }

    fn upload_initial_state(
        &mut self,
        model: &ServerMindTransformer,
        optimizer: &AdamWState,
    ) -> Result<(), String> {
        let p = self.total_params as usize;
        if optimizer.m.len() != p || optimizer.v.len() != p {
            return Err("Estado AdamW incompatível ao enviar para GPU.".to_string());
        }

        unsafe {
            let st = std::slice::from_raw_parts_mut(self.staging_mapped, self.staging_floats);
            let mut cur = 0usize;
            model.visit_param_slices(|slice| {
                st[cur..cur + slice.len()].copy_from_slice(slice);
                cur += slice.len();
            });
            assert_eq!(cur, p);
            st[p..2 * p].copy_from_slice(&optimizer.m);
            st[2 * p..3 * p].copy_from_slice(&optimizer.v);

            self.begin_cmd()?;
            // Zera toda a arena primeiro e em seguida copia pesos, `m` e `v`
            self.cmd_fill_zero(0, self.arena_total_floats as u32);
            self.cmd_barrier_compute_to_transfer();

            let copies = [
                VkBufferCopy {
                    src_offset: 0,
                    dst_offset: (self.off_weights as u64) * 4,
                    size: (p as u64) * 4,
                },
                VkBufferCopy {
                    src_offset: (p as u64) * 4,
                    dst_offset: (self.off_m as u64) * 4,
                    size: (p as u64) * 4,
                },
                VkBufferCopy {
                    src_offset: (2 * p as u64) * 4,
                    dst_offset: (self.off_v as u64) * 4,
                    size: (p as u64) * 4,
                },
            ];
            (self.dev.fns.cmd_copy_buffer)(
                self.cmd_buf,
                self.staging_buf,
                self.arena_buf,
                copies.len() as u32,
                copies.as_ptr(),
            );
            let mb = VkMemoryBarrier {
                s_type: VK_STYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            (self.dev.fns.cmd_pipeline_barrier)(
                self.cmd_buf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &mb,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            self.submit_and_wait()
        }
    }

    /// Baixa os pesos e momentos `m`/`v` atualizados da VRAM de volta para `model` e `optimizer` na RAM.
    pub fn sync_to_host(
        &mut self,
        model: &mut ServerMindTransformer,
        optimizer: &mut AdamWState,
    ) -> Result<(), String> {
        let p = self.total_params as usize;
        unsafe {
            self.begin_cmd()?;
            let pre_mb = VkMemoryBarrier {
                s_type: VK_STYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            };
            (self.dev.fns.cmd_pipeline_barrier)(
                self.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1,
                &pre_mb,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            let copies = [
                VkBufferCopy {
                    src_offset: (self.off_weights as u64) * 4,
                    dst_offset: 0,
                    size: (p as u64) * 4,
                },
                VkBufferCopy {
                    src_offset: (self.off_m as u64) * 4,
                    dst_offset: (p as u64) * 4,
                    size: (p as u64) * 4,
                },
                VkBufferCopy {
                    src_offset: (self.off_v as u64) * 4,
                    dst_offset: (2 * p as u64) * 4,
                    size: (p as u64) * 4,
                },
            ];
            (self.dev.fns.cmd_copy_buffer)(
                self.cmd_buf,
                self.arena_buf,
                self.staging_buf,
                copies.len() as u32,
                copies.as_ptr(),
            );
            let mb = VkMemoryBarrier {
                s_type: VK_STYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_HOST_READ_BIT,
            };
            (self.dev.fns.cmd_pipeline_barrier)(
                self.cmd_buf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                1,
                &mb,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            self.submit_and_wait()?;

            let st = std::slice::from_raw_parts(self.staging_mapped, self.staging_floats);
            let mut cur = 0usize;
            model.visit_param_slices_mut(|slice| {
                slice.copy_from_slice(&st[cur..cur + slice.len()]);
                cur += slice.len();
            });
            optimizer.m.copy_from_slice(&st[p..2 * p]);
            optimizer.v.copy_from_slice(&st[2 * p..3 * p]);
            Ok(())
        }
    }

    /// Baixa os gradientes acumulados na VRAM para `grads` (usado na verificação numérica contra a referência CPU).
    #[cfg(test)]
    pub fn download_grads(&mut self, grads: &mut super::TransformerGrads) -> Result<(), String> {
        let p = self.total_params as usize;
        unsafe {
            self.begin_cmd()?;
            let pre_mb = VkMemoryBarrier {
                s_type: VK_STYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            };
            (self.dev.fns.cmd_pipeline_barrier)(
                self.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1,
                &pre_mb,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            let copy = VkBufferCopy {
                src_offset: (self.off_grads as u64) * 4,
                dst_offset: 0,
                size: (p as u64) * 4,
            };
            (self.dev.fns.cmd_copy_buffer)(self.cmd_buf, self.arena_buf, self.staging_buf, 1, &copy);
            let post_mb = VkMemoryBarrier {
                s_type: VK_STYPE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_HOST_READ_BIT,
            };
            (self.dev.fns.cmd_pipeline_barrier)(
                self.cmd_buf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                1,
                &post_mb,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            self.submit_and_wait()?;

            let st = std::slice::from_raw_parts(self.staging_mapped, p);
            let mut cur = 0usize;
            let mut take = |dst: &mut [f32]| {
                dst.copy_from_slice(&st[cur..cur + dst.len()]);
                cur += dst.len();
            };
            take(&mut grads.token_emb);
            take(&mut grads.pos_emb);
            for b in &mut grads.blocks {
                take(&mut b.ln1_gamma);
                take(&mut b.wq);
                take(&mut b.wk);
                take(&mut b.wv);
                take(&mut b.wo);
                take(&mut b.ln2_gamma);
                take(&mut b.w1);
                take(&mut b.b1);
                take(&mut b.w2);
                take(&mut b.b2);
            }
            take(&mut grads.ln_f_gamma);
            take(&mut grads.lm_head);
            Ok(())
        }
    }

    /// Zera o vetor completo de gradientes acumulados na VRAM no início de um passo de treino.
    pub fn zero_grads(&mut self) -> Result<(), String> {
        unsafe {
            self.begin_cmd()?;
            self.cmd_fill_zero(self.off_grads, self.total_params);
            self.cmd_barrier_compute();
            self.submit_and_wait()
        }
    }

    /// Executa `forward + cross_entropy + backward` inteiramente na VRAM para uma janela `(input_ids, target_ids)`,
    /// transferindo apenas os IDs (`< 2 KiB`) e lendo de volta apenas o escalar `loss_sum` (`4` bytes).
    pub fn forward_and_backward_sample(
        &mut self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_id: usize,
    ) -> Result<(f32, usize), String> {
        let t_usize = input_ids
            .len()
            .min(target_ids.len())
            .min(self.config.max_seq_len);
        if t_usize == 0 {
            return Ok((0.0, 0));
        }
        let v = self.config.vocab_size;
        let valid_tokens = target_ids[..t_usize]
            .iter()
            .filter(|&&tgt| tgt != pad_id && tgt < v)
            .count();
        if valid_tokens == 0 {
            return Ok((0.0, 0));
        }

        unsafe {
            self.write_io_tokens(&input_ids[..t_usize], &target_ids[..t_usize]);
            self.begin_cmd()?;
            self.record_forward_and_loss(t_usize as u32, pad_id as u32, true);
            self.record_backward(t_usize as u32);
            self.cmd_barrier_shader_to_host();
            self.submit_and_wait()?;

            let loss_bits = *self.io_mapped.add(self.io_out_loss as usize);
            Ok((f32::from_bits(loss_bits), valid_tokens))
        }
    }

    /// Avalia a perda média de validação diretamente na VRAM (sem baixar pesos para a CPU).
    pub fn evaluate_loss(&mut self, windows: &[Vec<usize>], pad_id: usize) -> Result<f32, String> {
        if windows.is_empty() || self.config.vocab_size == 0 {
            return Ok(0.0);
        }
        let v = self.config.vocab_size;
        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        for win in windows {
            if win.len() < 2 {
                continue;
            }
            let inputs = &win[..win.len() - 1];
            let targets = &win[1..];
            let t_usize = inputs.len().min(targets.len()).min(self.config.max_seq_len);
            if t_usize == 0 {
                continue;
            }
            let valid_tokens = targets[..t_usize]
                .iter()
                .filter(|&&tgt| tgt != pad_id && tgt < v)
                .count();
            if valid_tokens == 0 {
                continue;
            }

            unsafe {
                self.write_io_tokens(&inputs[..t_usize], &targets[..t_usize]);
                self.begin_cmd()?;
                self.record_forward_and_loss(t_usize as u32, pad_id as u32, false);
                self.cmd_barrier_shader_to_host();
                self.submit_and_wait()?;

                let loss_sum = f32::from_bits(*self.io_mapped.add(self.io_out_loss as usize));
                total_loss += loss_sum as f64;
                total_tokens += valid_tokens;
            }
        }

        if total_tokens == 0 {
            Ok(0.0)
        } else {
            Ok((total_loss / total_tokens as f64) as f32)
        }
    }

    /// Avalia a perda média sobre sequências mascaradas (`inputs`, `targets` com `pad_id` nas perguntas) na VRAM.
    pub fn evaluate_masked_loss(
        &mut self,
        sequences: &[crate::training::MaskedSequence],
        pad_id: usize,
    ) -> Result<f32, String> {
        if sequences.is_empty() || self.config.vocab_size == 0 {
            return Ok(0.0);
        }
        let v = self.config.vocab_size;
        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        for seq in sequences {
            let t_usize = seq
                .inputs
                .len()
                .min(seq.targets.len())
                .min(self.config.max_seq_len);
            if t_usize == 0 {
                continue;
            }
            let valid_tokens = seq.targets[..t_usize]
                .iter()
                .filter(|&&tgt| tgt != pad_id && tgt < v)
                .count();
            if valid_tokens == 0 {
                continue;
            }
            unsafe {
                self.write_io_tokens(&seq.inputs[..t_usize], &seq.targets[..t_usize]);
                self.begin_cmd()?;
                self.record_forward_and_loss(t_usize as u32, pad_id as u32, false);
                self.cmd_barrier_shader_to_host();
                self.submit_and_wait()?;

                let loss_sum = f32::from_bits(*self.io_mapped.add(self.io_out_loss as usize));
                total_loss += loss_sum as f64;
                total_tokens += valid_tokens;
            }
        }

        if total_tokens == 0 {
            Ok(0.0)
        } else {
            Ok((total_loss / total_tokens as f64) as f32)
        }
    }

    /// Escala gradientes (`1 / step_tokens`), calcula `grad_norm`, valida finitude (`SM-001`) e
    /// executa o passo `AdamW` completo diretamente na VRAM da GPU.
    pub fn step_adamw(
        &mut self,
        optimizer_step: &mut usize,
        step_tokens: usize,
        lr: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> Result<Option<f32>, String> {
        if step_tokens == 0 {
            return Ok(Some(0.0));
        }
        if !lr.is_finite() || !weight_decay.is_finite() {
            return Ok(None);
        }

        let inv_tokens = 1.0f32 / (step_tokens as f32);
        let num_wgs = ((self.total_params as usize + 255) / 256) as u32;

        // 1. Escala `grads` por `1 / step_tokens` e reduz `sum(grads^2)` em `io_out_grad_sq`
        let grad_norm = unsafe {
            self.begin_cmd()?;
            let mut pc = [0u32; PUSH_CONSTANT_WORDS];
            pc[0] = self.off_grads;
            pc[1] = self.total_params;
            pc[2] = self.ws_grad_partial;
            pc[3] = inv_tokens.to_bits();
            self.dispatch_1d(self.dev.pipelines.scale_and_grad_sq, &pc, num_wgs);
            self.cmd_barrier_compute();

            let mut rpc = [0u32; PUSH_CONSTANT_WORDS];
            rpc[0] = self.ws_grad_partial;
            rpc[1] = num_wgs;
            rpc[2] = self.io_out_grad_sq;
            rpc[3] = 0;
            self.dispatch_1d(self.dev.pipelines.reduce_sum_f32_to_io, &rpc, 1);
            self.cmd_barrier_shader_to_host();
            self.submit_and_wait()?;

            let sq_bits = *self.io_mapped.add(self.io_out_grad_sq as usize);
            let sq = f32::from_bits(sq_bits);
            if !sq.is_finite() || sq < 0.0 {
                return Ok(None);
            }
            sq.sqrt()
        };

        if !grad_norm.is_finite() {
            return Ok(None);
        }

        // 2. Aplica clipping + AdamW por grupo de decaimento sem tocar a RAM
        let clip_scale = if max_grad_norm.is_finite() && max_grad_norm > 0.0 && grad_norm > max_grad_norm {
            max_grad_norm / grad_norm.max(1e-8)
        } else {
            1.0f32
        };

        *optimizer_step += 1;
        let step_i32 = (*optimizer_step).min(i32::MAX as usize) as i32;
        let bc1 = (1.0 - 0.9f32.powi(step_i32)).max(1e-8);
        let bc2 = (1.0 - 0.999f32.powi(step_i32)).max(1e-12);

        unsafe {
            self.begin_cmd()?;
            let groups = self.adam_groups.clone();
            for grp in groups {
                let wd_eff = if grp.apply_decay && weight_decay > 0.0 {
                    lr * weight_decay
                } else {
                    0.0
                };
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = self.off_weights + grp.rel_off;
                pc[1] = self.off_grads + grp.rel_off;
                pc[2] = self.off_m + grp.rel_off;
                pc[3] = self.off_v + grp.rel_off;
                pc[4] = grp.len;
                pc[5] = lr.to_bits();
                pc[6] = wd_eff.to_bits();
                pc[7] = bc1.to_bits();
                pc[8] = bc2.to_bits();
                pc[9] = clip_scale.to_bits();
                let wgs = (grp.len + 255) / 256;
                self.dispatch_1d(self.dev.pipelines.adamw_step, &pc, wgs);
            }
            self.cmd_barrier_compute();
            self.submit_and_wait()?;
        }

        Ok(Some(grad_norm))
    }

    // --- Helpers Internos de Gravação do Grafo Vulkan ---

    unsafe fn write_io_tokens(&mut self, inputs: &[usize], targets: &[usize]) {
        let t = inputs.len();
        for i in 0..t {
            *self.io_mapped.add(self.io_in_off as usize + i) = inputs[i] as u32;
            *self.io_mapped.add(self.io_tgt_off as usize + i) = targets[i] as u32;
        }
        *self.io_mapped.add(self.io_out_loss as usize) = 0u32;
    }

    unsafe fn begin_cmd(&self) -> Result<(), String> {
        let _ = (self.dev.fns.reset_command_buffer)(self.cmd_buf, 0);
        let begin = VkCommandBufferBeginInfo {
            s_type: VK_STYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        let res = (self.dev.fns.begin_command_buffer)(self.cmd_buf, &begin);
        if res != VK_SUCCESS {
            return Err(format!("vkBeginCommandBuffer falhou ({}).", res));
        }
        (self.dev.fns.cmd_bind_descriptor_sets)(
            self.cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.dev.pipeline_layout,
            0,
            1,
            &self.desc_set,
            0,
            ptr::null(),
        );
        Ok(())
    }

    unsafe fn submit_and_wait(&self) -> Result<(), String> {
        let res = (self.dev.fns.end_command_buffer)(self.cmd_buf);
        if res != VK_SUCCESS {
            return Err(format!("vkEndCommandBuffer falhou ({}).", res));
        }
        let q = self
            .dev
            .queue_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = (self.dev.fns.reset_fences)(self.dev.device, 1, &self.fence);
        let sub = VkSubmitInfo {
            s_type: VK_STYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.cmd_buf,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        let sres = (self.dev.fns.queue_submit)(*q, 1, &sub, self.fence);
        if sres != VK_SUCCESS {
            return Err(format!("vkQueueSubmit falhou ({}).", sres));
        }
        let wres = (self.dev.fns.wait_for_fences)(self.dev.device, 1, &self.fence, 1, u64::MAX);
        if wres != VK_SUCCESS {
            return Err(format!("vkWaitForFences falhou ({}).", wres));
        }
        Ok(())
    }

    #[inline]
    unsafe fn dispatch_3d(
        &self,
        pipeline: VkPipeline,
        pc: &[u32; PUSH_CONSTANT_WORDS],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        if gx == 0 || gy == 0 || gz == 0 {
            return;
        }
        (self.dev.fns.cmd_bind_pipeline)(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        (self.dev.fns.cmd_push_constants)(
            self.cmd_buf,
            self.dev.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            PUSH_CONSTANT_BYTES,
            pc.as_ptr() as *const c_void,
        );
        (self.dev.fns.cmd_dispatch)(self.cmd_buf, gx, gy, gz);
    }

    #[inline]
    unsafe fn dispatch_1d(&self, pipeline: VkPipeline, pc: &[u32; PUSH_CONSTANT_WORDS], gx: u32) {
        self.dispatch_3d(pipeline, pc, gx, 1, 1);
    }

    #[inline]
    unsafe fn cmd_barrier_compute(&self) {
        let mb = VkMemoryBarrier {
            s_type: VK_STYPE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
            dst_access_mask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        (self.dev.fns.cmd_pipeline_barrier)(
            self.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &mb,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
    }

    #[inline]
    unsafe fn cmd_barrier_compute_to_transfer(&self) {
        let mb = VkMemoryBarrier {
            s_type: VK_STYPE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
            dst_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        };
        (self.dev.fns.cmd_pipeline_barrier)(
            self.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1,
            &mb,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
    }

    #[inline]
    unsafe fn cmd_barrier_shader_to_host(&self) {
        let mb = VkMemoryBarrier {
            s_type: VK_STYPE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
            dst_access_mask: VK_ACCESS_HOST_READ_BIT,
        };
        (self.dev.fns.cmd_pipeline_barrier)(
            self.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            &mb,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
    }

    #[inline]
    unsafe fn cmd_fill_zero(&self, off_dst: u32, len: u32) {
        let mut pc = [0u32; PUSH_CONSTANT_WORDS];
        pc[0] = off_dst;
        pc[1] = len;
        pc[2] = 0.0f32.to_bits();
        self.dispatch_1d(self.dev.pipelines.fill_f32, &pc, (len + 255) / 256);
    }

    #[inline]
    unsafe fn cmd_copy_arena_slice(&self, dst_off: u32, src_off: u32, len: u32) {
        let region = VkBufferCopy {
            src_offset: (src_off as u64) * 4,
            dst_offset: (dst_off as u64) * 4,
            size: (len as u64) * 4,
        };
        let pre = VkMemoryBarrier {
            s_type: VK_STYPE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
            dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
        };
        (self.dev.fns.cmd_pipeline_barrier)(
            self.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1,
            &pre,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
        (self.dev.fns.cmd_copy_buffer)(self.cmd_buf, self.arena_buf, self.arena_buf, 1, &region);
        let post = VkMemoryBarrier {
            s_type: VK_STYPE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
            dst_access_mask: VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        (self.dev.fns.cmd_pipeline_barrier)(
            self.cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &post,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
    }

    #[inline]
    unsafe fn cmd_matmul_nn(&self, off_a: u32, off_b: u32, off_c: u32, m: u32, k: u32, n: u32) {
        let mut pc = [0u32; PUSH_CONSTANT_WORDS];
        pc[0] = off_a;
        pc[1] = off_b;
        pc[2] = off_c;
        pc[3] = m;
        pc[4] = k;
        pc[5] = n;
        self.dispatch_3d(
            self.dev.pipelines.matmul_nn,
            &pc,
            (n + 15) / 16,
            (m + 15) / 16,
            1,
        );
    }

    #[inline]
    unsafe fn cmd_matmul_tn(&self, off_a: u32, off_b: u32, off_c: u32, k: u32, m: u32, n: u32) {
        let mut pc = [0u32; PUSH_CONSTANT_WORDS];
        pc[0] = off_a;
        pc[1] = off_b;
        pc[2] = off_c;
        pc[3] = k;
        pc[4] = m;
        pc[5] = n;
        self.dispatch_3d(
            self.dev.pipelines.matmul_tn,
            &pc,
            (n + 15) / 16,
            (k + 15) / 16,
            1,
        );
    }

    #[inline]
    unsafe fn cmd_matmul_nt(&self, off_a: u32, off_b: u32, off_c: u32, m: u32, n: u32, k: u32) {
        let mut pc = [0u32; PUSH_CONSTANT_WORDS];
        pc[0] = off_a;
        pc[1] = off_b;
        pc[2] = off_c;
        pc[3] = m;
        pc[4] = n;
        pc[5] = k;
        self.dispatch_3d(
            self.dev.pipelines.matmul_nt,
            &pc,
            (k + 15) / 16,
            (m + 15) / 16,
            1,
        );
    }

    unsafe fn record_forward_and_loss(&self, t: u32, pad_id: u32, write_dlogits: bool) {
        let d = self.config.d_model as u32;
        let ff = self.config.d_ff as u32;
        let v = self.config.vocab_size as u32;
        let n_heads = self.config.n_heads as u32;
        let scale = 1.0f32 / (self.config.head_dim() as f32).sqrt();
        let td = t * d;
        let tff = t * ff;
        let tv = t * v;

        // 1. Embeddings forward -> `ws_x`
        {
            let mut pc = [0u32; PUSH_CONSTANT_WORDS];
            pc[0] = self.off_weights + self.rel_tok_emb;
            pc[1] = self.off_weights + self.rel_pos_emb;
            pc[2] = self.ws_x;
            pc[3] = t;
            pc[4] = d;
            pc[5] = v;
            pc[6] = self.io_in_off;
            self.dispatch_1d(self.dev.pipelines.embed_fwd, &pc, (td + 255) / 256);
            self.cmd_barrier_compute();
        }

        // 2. Camadas Transformer
        for l in 0..self.config.n_layers {
            let wp = self.rel_layers[l];
            let ac = self.act_layers[l];

            // Salva `x_in = ws_x`
            self.cmd_copy_arena_slice(ac.x_in, self.ws_x, td);

            // RMSNorm 1
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.x_in;
                pc[1] = self.off_weights + wp.ln1_gamma;
                pc[2] = ac.ln1;
                pc[3] = ac.inv_rms1;
                pc[4] = t;
                pc[5] = d;
                self.dispatch_1d(self.dev.pipelines.rmsnorm_fwd, &pc, t);
            }
            // Zera Q, K, V e attn_out em paralelo com/após RMSNorm
            self.cmd_fill_zero(ac.q, td);
            self.cmd_fill_zero(ac.k, td);
            self.cmd_fill_zero(ac.v, td);
            self.cmd_fill_zero(self.ws_attn_out, td);
            self.cmd_barrier_compute();

            // Projeções Q, K, V
            self.cmd_matmul_nn(ac.ln1, self.off_weights + wp.wq, ac.q, t, d, d);
            self.cmd_matmul_nn(ac.ln1, self.off_weights + wp.wk, ac.k, t, d, d);
            self.cmd_matmul_nn(ac.ln1, self.off_weights + wp.wv, ac.v, t, d, d);
            self.cmd_barrier_compute();

            // Multi-Head Causal Self-Attention
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.q;
                pc[1] = ac.k;
                pc[2] = ac.v;
                pc[3] = ac.attn_probs;
                pc[4] = ac.attn_ctx;
                pc[5] = t;
                pc[6] = d;
                pc[7] = n_heads;
                pc[8] = scale.to_bits();
                self.dispatch_1d(
                    self.dev.pipelines.causal_mha_fwd,
                    &pc,
                    (n_heads * t + 63) / 64,
                );
                self.cmd_barrier_compute();
            }

            // Projeção Wo + Residual 1 (`x_mid = x_in + attn_out`)
            self.cmd_matmul_nn(
                ac.attn_ctx,
                self.off_weights + wp.wo,
                self.ws_attn_out,
                t,
                d,
                d,
            );
            self.cmd_barrier_compute();

            self.cmd_copy_arena_slice(ac.x_mid, ac.x_in, td);
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.x_mid;
                pc[1] = self.ws_attn_out;
                pc[2] = td;
                self.dispatch_1d(self.dev.pipelines.vec_add_inplace, &pc, (td + 255) / 256);
            }
            self.cmd_fill_zero(ac.h_pre, tff);
            self.cmd_fill_zero(self.ws_mlp_out, td);
            self.cmd_barrier_compute();

            // RMSNorm 2
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.x_mid;
                pc[1] = self.off_weights + wp.ln2_gamma;
                pc[2] = ac.ln2;
                pc[3] = ac.inv_rms2;
                pc[4] = t;
                pc[5] = d;
                self.dispatch_1d(self.dev.pipelines.rmsnorm_fwd, &pc, t);
                self.cmd_barrier_compute();
            }

            // MLP W1 + b1 + GELU -> `h_act`
            self.cmd_matmul_nn(ac.ln2, self.off_weights + wp.w1, ac.h_pre, t, d, ff);
            self.cmd_barrier_compute();
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.h_pre;
                pc[1] = self.off_weights + wp.b1;
                pc[2] = ac.h_act;
                pc[3] = tff;
                pc[4] = ff;
                self.dispatch_1d(self.dev.pipelines.bias_gelu_fwd, &pc, (tff + 255) / 256);
                self.cmd_barrier_compute();
            }

            // MLP W2 + b2 + Residual 2 (`ws_x = x_mid + mlp_out + b2`)
            self.cmd_matmul_nn(ac.h_act, self.off_weights + wp.w2, self.ws_mlp_out, t, ff, d);
            self.cmd_barrier_compute();
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = ac.x_mid;
                pc[1] = self.ws_mlp_out;
                pc[2] = self.off_weights + wp.b2;
                pc[3] = self.ws_x;
                pc[4] = td;
                pc[5] = d;
                self.dispatch_1d(self.dev.pipelines.bias_res_add_fwd, &pc, (td + 255) / 256);
                self.cmd_barrier_compute();
            }
        }

        // 3. RMSNorm final + LM Head
        {
            let mut pc = [0u32; PUSH_CONSTANT_WORDS];
            pc[0] = self.ws_x;
            pc[1] = self.off_weights + self.rel_ln_f_gamma;
            pc[2] = self.ws_ln_f;
            pc[3] = self.ws_inv_rms_f;
            pc[4] = t;
            pc[5] = d;
            self.dispatch_1d(self.dev.pipelines.rmsnorm_fwd, &pc, t);
        }
        self.cmd_fill_zero(self.ws_logits, tv);
        self.cmd_barrier_compute();

        self.cmd_matmul_nn(
            self.ws_ln_f,
            self.off_weights + self.rel_lm_head,
            self.ws_logits,
            t,
            d,
            v,
        );
        self.cmd_barrier_compute();

        // 4. Cross-Entropy + Redução da soma de perdas para `io_out_loss`
        {
            let mut pc = [0u32; PUSH_CONSTANT_WORDS];
            pc[0] = self.ws_logits;
            pc[1] = self.ws_d_logits;
            pc[2] = self.ws_row_losses;
            pc[3] = v;
            pc[4] = pad_id;
            pc[5] = self.io_tgt_off;
            pc[6] = u32::from(write_dlogits);
            self.dispatch_1d(self.dev.pipelines.cross_entropy_fwd_bwd, &pc, t);
            self.cmd_barrier_compute();

            let mut rpc = [0u32; PUSH_CONSTANT_WORDS];
            rpc[0] = self.ws_row_losses;
            rpc[1] = t;
            rpc[2] = self.io_out_loss;
            rpc[3] = 0;
            self.dispatch_1d(self.dev.pipelines.reduce_sum_f32_to_io, &rpc, 1);
            self.cmd_barrier_compute();
        }
    }

    unsafe fn record_backward(&self, t: u32) {
        let d = self.config.d_model as u32;
        let ff = self.config.d_ff as u32;
        let v = self.config.vocab_size as u32;
        let n_heads = self.config.n_heads as u32;
        let scale = 1.0f32 / (self.config.head_dim() as f32).sqrt();
        let td = t * d;
        let tff = t * ff;

        // 1. Backprop LM Head + RMSNorm final
        self.cmd_fill_zero(self.ws_d_ln_f, td);
        self.cmd_barrier_compute();

        self.cmd_matmul_tn(
            self.ws_ln_f,
            self.ws_d_logits,
            self.off_grads + self.rel_lm_head,
            d,
            t,
            v,
        );
        self.cmd_matmul_nt(
            self.ws_d_logits,
            self.off_weights + self.rel_lm_head,
            self.ws_d_ln_f,
            t,
            v,
            d,
        );
        self.cmd_barrier_compute();

        self.cmd_rmsnorm_bwd(
            self.ws_x,
            self.off_weights + self.rel_ln_f_gamma,
            self.ws_inv_rms_f,
            self.ws_d_ln_f,
            self.off_grads + self.rel_ln_f_gamma,
            self.ws_d_x,
            t,
            d,
        );
        self.cmd_barrier_compute();

        // 2. Backprop através dos blocos em ordem reversa
        for l in (0..self.config.n_layers).rev() {
            let wp = self.rel_layers[l];
            let ac = self.act_layers[l];

            // `d_x_mid = d_x`
            self.cmd_copy_arena_slice(self.ws_d_x_mid, self.ws_d_x, td);
            self.cmd_fill_zero(self.ws_d_h_act, tff);
            self.cmd_fill_zero(self.ws_d_ln2, td);
            self.cmd_barrier_compute();

            // `b2`, `w2`, `d_h_act`
            self.cmd_bias_grad(self.ws_d_x, self.off_grads + wp.b2, t, d);
            self.cmd_matmul_tn(ac.h_act, self.ws_d_x, self.off_grads + wp.w2, ff, t, d);
            self.cmd_matmul_nt(
                self.ws_d_x,
                self.off_weights + wp.w2,
                self.ws_d_h_act,
                t,
                d,
                ff,
            );
            self.cmd_barrier_compute();

            // `gelu_bwd` -> `d_h_pre`
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = self.ws_d_h_act;
                pc[1] = ac.h_pre;
                pc[2] = self.ws_d_h_pre;
                pc[3] = tff;
                self.dispatch_1d(self.dev.pipelines.gelu_bwd, &pc, (tff + 255) / 256);
                self.cmd_barrier_compute();
            }

            // `b1`, `w1`, `d_ln2`
            self.cmd_bias_grad(self.ws_d_h_pre, self.off_grads + wp.b1, t, ff);
            self.cmd_matmul_tn(ac.ln2, self.ws_d_h_pre, self.off_grads + wp.w1, d, t, ff);
            self.cmd_matmul_nt(
                self.ws_d_h_pre,
                self.off_weights + wp.w1,
                self.ws_d_ln2,
                t,
                ff,
                d,
            );
            self.cmd_barrier_compute();

            // `rmsnorm2_bwd` + soma residual `d_x_mid += d_rms_tmp`
            self.cmd_rmsnorm_bwd(
                ac.x_mid,
                self.off_weights + wp.ln2_gamma,
                ac.inv_rms2,
                self.ws_d_ln2,
                self.off_grads + wp.ln2_gamma,
                self.ws_d_rms_tmp,
                t,
                d,
            );
            self.cmd_barrier_compute();
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = self.ws_d_x_mid;
                pc[1] = self.ws_d_rms_tmp;
                pc[2] = td;
                self.dispatch_1d(self.dev.pipelines.vec_add_inplace, &pc, (td + 255) / 256);
                self.cmd_barrier_compute();
            }

            // `d_x_in = d_x_mid`, `wo`, `d_attn_ctx`
            self.cmd_copy_arena_slice(self.ws_d_x_in, self.ws_d_x_mid, td);
            self.cmd_fill_zero(self.ws_d_attn_ctx, td);
            self.cmd_barrier_compute();

            self.cmd_matmul_tn(
                ac.attn_ctx,
                self.ws_d_x_mid,
                self.off_grads + wp.wo,
                d,
                t,
                d,
            );
            self.cmd_matmul_nt(
                self.ws_d_x_mid,
                self.off_weights + wp.wo,
                self.ws_d_attn_ctx,
                t,
                d,
                d,
            );
            self.cmd_barrier_compute();

            // Atenção Backward em 2 passos 100% livres de corridas
            {
                let mut pc1 = [0u32; PUSH_CONSTANT_WORDS];
                pc1[0] = ac.k;
                pc1[1] = ac.v;
                pc1[2] = ac.attn_probs;
                pc1[3] = self.ws_d_attn_ctx;
                pc1[4] = self.ws_d_scores;
                pc1[5] = self.ws_d_q;
                pc1[6] = t;
                pc1[7] = d;
                pc1[8] = n_heads;
                pc1[9] = scale.to_bits();
                self.dispatch_1d(
                    self.dev.pipelines.causal_mha_bwd_q_ds,
                    &pc1,
                    (n_heads * t + 63) / 64,
                );
                self.cmd_barrier_compute();

                let mut pc2 = [0u32; PUSH_CONSTANT_WORDS];
                pc2[0] = ac.q;
                pc2[1] = ac.attn_probs;
                pc2[2] = self.ws_d_attn_ctx;
                pc2[3] = self.ws_d_scores;
                pc2[4] = self.ws_d_k;
                pc2[5] = self.ws_d_v;
                pc2[6] = t;
                pc2[7] = d;
                pc2[8] = n_heads;
                self.dispatch_1d(
                    self.dev.pipelines.causal_mha_bwd_kv,
                    &pc2,
                    (td + 63) / 64,
                );
            }
            self.cmd_fill_zero(self.ws_d_ln1, td);
            self.cmd_barrier_compute();

            // `wq`, `wk`, `wv` e acumulação em `d_ln1`
            self.cmd_matmul_tn(ac.ln1, self.ws_d_q, self.off_grads + wp.wq, d, t, d);
            self.cmd_matmul_tn(ac.ln1, self.ws_d_k, self.off_grads + wp.wk, d, t, d);
            self.cmd_matmul_tn(ac.ln1, self.ws_d_v, self.off_grads + wp.wv, d, t, d);

            self.cmd_matmul_nt(
                self.ws_d_q,
                self.off_weights + wp.wq,
                self.ws_d_ln1,
                t,
                d,
                d,
            );
            self.cmd_barrier_compute();
            self.cmd_matmul_nt(
                self.ws_d_k,
                self.off_weights + wp.wk,
                self.ws_d_ln1,
                t,
                d,
                d,
            );
            self.cmd_barrier_compute();
            self.cmd_matmul_nt(
                self.ws_d_v,
                self.off_weights + wp.wv,
                self.ws_d_ln1,
                t,
                d,
                d,
            );
            self.cmd_barrier_compute();

            // `rmsnorm1_bwd` + residual `d_x = d_x_in + d_rms_tmp`
            self.cmd_rmsnorm_bwd(
                ac.x_in,
                self.off_weights + wp.ln1_gamma,
                ac.inv_rms1,
                self.ws_d_ln1,
                self.off_grads + wp.ln1_gamma,
                self.ws_d_rms_tmp,
                t,
                d,
            );
            self.cmd_barrier_compute();
            self.cmd_copy_arena_slice(self.ws_d_x, self.ws_d_x_in, td);
            {
                let mut pc = [0u32; PUSH_CONSTANT_WORDS];
                pc[0] = self.ws_d_x;
                pc[1] = self.ws_d_rms_tmp;
                pc[2] = td;
                self.dispatch_1d(self.dev.pipelines.vec_add_inplace, &pc, (td + 255) / 256);
                self.cmd_barrier_compute();
            }
        }

        // 3. Embeddings backward
        {
            let mut pc = [0u32; PUSH_CONSTANT_WORDS];
            pc[0] = self.ws_d_x;
            pc[1] = self.off_grads + self.rel_tok_emb;
            pc[2] = self.off_grads + self.rel_pos_emb;
            pc[3] = t;
            pc[4] = d;
            pc[5] = v;
            pc[6] = self.io_in_off;
            self.dispatch_1d(self.dev.pipelines.embed_bwd, &pc, (d + 63) / 64);
            self.cmd_barrier_compute();
        }
    }

    #[inline]
    unsafe fn cmd_rmsnorm_bwd(
        &self,
        off_x: u32,
        off_gamma: u32,
        off_inv: u32,
        off_dout: u32,
        off_dgamma: u32,
        off_dx: u32,
        t: u32,
        d: u32,
    ) {
        let mut pc_dx = [0u32; PUSH_CONSTANT_WORDS];
        pc_dx[0] = off_x;
        pc_dx[1] = off_gamma;
        pc_dx[2] = off_inv;
        pc_dx[3] = off_dout;
        pc_dx[4] = off_dx;
        pc_dx[5] = d;
        self.dispatch_1d(self.dev.pipelines.rmsnorm_bwd_dx, &pc_dx, t);

        let mut pc_dg = [0u32; PUSH_CONSTANT_WORDS];
        pc_dg[0] = off_x;
        pc_dg[1] = off_inv;
        pc_dg[2] = off_dout;
        pc_dg[3] = off_dgamma;
        pc_dg[4] = t;
        pc_dg[5] = d;
        self.dispatch_1d(self.dev.pipelines.rmsnorm_bwd_dgamma, &pc_dg, (d + 63) / 64);
    }

    #[inline]
    unsafe fn cmd_bias_grad(&self, off_dmat: u32, off_dbias: u32, t: u32, cols: u32) {
        let mut pc = [0u32; PUSH_CONSTANT_WORDS];
        pc[0] = off_dmat;
        pc[1] = off_dbias;
        pc[2] = t;
        pc[3] = cols;
        self.dispatch_1d(self.dev.pipelines.bias_grad_sum, &pc, (cols + 63) / 64);
    }
}
