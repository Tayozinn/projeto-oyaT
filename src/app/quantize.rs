use std::path::Path;

pub(crate) struct QuantExportStats {
    pub(crate) precision: crate::neural::WeightPrecision,
    pub(crate) src_disk_mb: f64,
    pub(crate) dst_disk_mb: f64,
    pub(crate) src_ram_mb: f64,
    pub(crate) dst_ram_mb: f64,
    pub(crate) compression_ratio: f64,
}

pub(crate) fn export_quantized_package(
    src_dir: &Path,
    dst_dir: &Path,
    precision: crate::neural::WeightPrecision,
) -> Result<QuantExportStats, String> {
    let tok_src = src_dir.join("tokenizer.json");
    if !tok_src.is_file() {
        return Err(format!(
            "Tokenizer '{}' não encontrado no diretório de origem.",
            tok_src.display()
        ));
    }
    let best_path = src_dir.join("best.bin");
    let latest_path = src_dir.join("latest.bin");
    let src_ckpt = if best_path.is_file() {
        best_path
    } else if latest_path.is_file() {
        latest_path
    } else {
        return Err(format!(
            "Nenhum checkpoint FP32 ('best.bin' ou 'latest.bin') encontrado em '{}'.",
            src_dir.display()
        ));
    };

    let src_disk_bytes = std::fs::metadata(&src_ckpt)
        .map_err(|e| e.to_string())?
        .len();
    let (fp32_model, _opt, meta) = crate::neural::load_checkpoint(&src_ckpt)?;
    let src_ram_bytes = fp32_model.actual_weight_bytes();

    let qmodel = crate::neural::quantize_model_in_memory(&fp32_model, precision)?;
    let dst_ram_bytes = qmodel.actual_weight_bytes();

    std::fs::create_dir_all(dst_dir).map_err(|e| {
        format!(
            "Falha ao criar pasta destino '{}': {}",
            dst_dir.display(),
            e
        )
    })?;
    let tok_dst = dst_dir.join("tokenizer.json");
    if src_dir != dst_dir {
        std::fs::copy(&tok_src, &tok_dst).map_err(|e| {
            format!(
                "Falha ao copiar tokenizer para '{}': {}",
                tok_dst.display(),
                e
            )
        })?;
    }
    let smq_path = dst_dir.join("quantized.smq");
    crate::neural::save_quantized_checkpoint(&smq_path, &qmodel, meta.step, meta.best_val_loss)?;
    let dst_disk_bytes = std::fs::metadata(&smq_path)
        .map_err(|e| e.to_string())?
        .len();

    let mb = 1024.0 * 1024.0;
    let src_disk_mb = src_disk_bytes as f64 / mb;
    let dst_disk_mb = dst_disk_bytes as f64 / mb;
    Ok(QuantExportStats {
        precision,
        src_disk_mb,
        dst_disk_mb,
        src_ram_mb: src_ram_bytes as f64 / mb,
        dst_ram_mb: dst_ram_bytes as f64 / mb,
        compression_ratio: src_disk_mb / dst_disk_mb.max(1e-9),
    })
}
