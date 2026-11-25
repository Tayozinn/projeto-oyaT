# SM-006 — Backend GPU Modular Vulkan Compute para AMD Radeon RX 7600

## 1. Contexto de Hardware e Decisões Arquiteturais
- **GPU Alvo de Desenvolvimento:** AMD Radeon RX 7600 (`vendorID = 0x1002`, `deviceID = 0x7480`, RDNA 3 / `gfx1102`, `8 GB` GDDR6, `Vulkan 1.4.359`, driver `AMD proprietary 26.11.2 (LLPC)`).
- **Restrições Reais Verificadas:**
  1. `ROCm` não é viável para `gfx1102` neste ambiente Windows;
  2. `VULKAN_SDK` e `glslc` não estão instalados na máquina;
  3. O link entre Intel Core i5-9600K e RX 7600 opera em **`PCIe 3.0 x8`** (~`5,5–6,5 GB/s`), o que tornaria o offload isolado de MatMul mais lento que ou empatado com o backend CPU `AVX2+FMA` devido à latência de cópia host↔device;
  4. O binário distribuído deve continuar executando normalmente em computadores modestos de `8 GB` de RAM **sem GPU dedicada** e sem `vulkan-1.dll`.

## 2. Arquitetura Implementada

### 2.1. Carregamento Dinâmico Zero-SDK (`src/neural/gpu.rs`)
- Carrega dinamicamente `vulkan-1.dll` (Windows) ou `libvulkan.so.1` (Linux) em tempo de execução via `LoadLibraryA`/`GetProcAddress` e `dlopen`/`dlsym`, sem link estático de compilação e sem adicionar nenhuma crate externa.
- Quando nenhuma biblioteca ou GPU compatível com Vulkan 1.1 Compute está presente, `VulkanDevice::global()` retorna `Err(...)` limpamente e o sistema opera 100% em CPU (`CpuThreadPool` + `AVX2+FMA` / `Scalar-Tiled`).

### 2.2. Gerador SPIR-V 1.3 Nativo em Rust Puro (`src/neural/spirv.rs`)
- Constrói em memória os **21 shaders Compute** necessários para o ciclo completo do Transformer Decoder-only obedecendo estritamente ao layout lógico Khronos SPIR-V 1.3 (`0x00010300`):
  - Utilitários e Embeddings: `fill_f32`, `vec_add_inplace`, `embed_fwd`, `embed_bwd` (acumulação por coluna livre de corridas sem float atomics);
  - Normalização: `rmsnorm_fwd`, `rmsnorm_bwd_dx` (redução em árvore em LDS de `64` lanes) e `rmsnorm_bwd_dgamma`;
  - Álgebra Linear em LDS (`Workgroup` Shared Memory `16×16`): `matmul_nn_tiled`, `matmul_tn_tiled` (leitura coalescida com transposição em LDS) e `matmul_nt_tiled`;
  - Atenção Causal Multi-Cabeça: `causal_mha_fwd`, `causal_mha_bwd_q_ds` e `causal_mha_bwd_kv` (divisão em 2 passos 100% livres de condições de corrida em `d_k` e `d_v`);
  - MLP & Perda: `bias_gelu_fwd`, `gelu_bwd` (preservando saturação exata `|x| >= 10.0`), `bias_res_add_fwd`, `bias_grad_sum`, `cross_entropy_fwd_bwd` (`log-sum-exp` paralelo por token) e `reduce_sum_f32_to_io`;
  - Otimizador: `scale_and_grad_sq` e `adamw_step` (respeitando exatamente os grupos `apply_decay` seletivos do `AdamWState::step` em CPU).

### 2.3. Execução Residente em VRAM (`VulkanTrainingSession`)
- Sub-aloca **todos** os pesos, gradientes, momentos `m`/`v` do AdamW, caches de ativação por camada e tensores temporários em **uma única Arena `DEVICE_LOCAL` na VRAM de 8 GB da RX 7600**, vinculada uma única vez em `DescriptorSet 0, Binding 0`.
- Durante cada amostra de treinamento:
  - Apenas os IDs `input_ids` e `target_ids` (`< 2 KiB`) são escritos no buffer coerente `io_buf`;
  - Todo o grafo de `forward + cross_entropy + backward` de todas as camadas é gravado e submetido em **um único `VkCommandBuffer`** sincronizado internamente em VRAM (`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`);
  - Apenas `loss_sum` (`4` bytes) e `grad_norm_sq` (`4` bytes) retornam à CPU por passo!
- O download completo de pesos + estados AdamW (`sync_to_host`) ocorre **somente** quando um checkpoint (`best.bin` / `latest.bin`) precisa ser persistido ou ao encerrar o treinamento, gerando arquivos `SMNN` v2 100% idênticos e compatíveis com inferência CPU.

## 3. Seleção de Dispositivo (`--device auto|cpu|gpu`)
- `--device cpu`: força execução em CPU (`CpuThreadPool` + `AVX2+FMA`).
- `--device gpu`: força execução na GPU Vulkan (`AMD Radeon RX 7600`), falhando explicitamente caso indisponível.
- `--device auto` *(padrão)*: seleciona automaticamente a GPU discreta Vulkan quando `total_params >= GPU_AUTO_MIN_PARAMS`, mantendo modelos pequenos na CPU `AVX2+FMA` onde a latência sub-milissegundo sem despacho de fila é vantajosa.
