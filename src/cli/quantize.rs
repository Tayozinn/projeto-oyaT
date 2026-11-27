pub(super) struct QuantExportStats {
    pub(super) precision: crate::neural::WeightPrecision,
    pub(super) src_disk_mb: f64,
    pub(super) dst_disk_mb: f64,
    pub(super) src_ram_mb: f64,
    pub(super) dst_ram_mb: f64,
    pub(super) compression_ratio: f64,
}

pub(super) fn export_quantized_package(
    src_dir: &std::path::Path,
    dst_dir: &std::path::Path,
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

pub(super) fn handle_quantize_command(arg: &str) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() < 2 {
        println!("Uso: :quantize <pasta_ckpt_fp32> <pasta_saida_smq> [fp16|bf16|int8|int4]");
        return;
    }
    let bits_raw = parts.get(2).copied().unwrap_or("int8");
    let prec = match crate::neural::WeightPrecision::parse(bits_raw) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match export_quantized_package(
        std::path::Path::new(parts[0]),
        std::path::Path::new(parts[1]),
        prec,
    ) {
        Ok(st) => {
            println!(
                "Pacote SMQ1 ({}) gravado em '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | RAM pesos {:.2} MB -> {:.2} MB.",
                st.precision.as_str(),
                parts[1],
                st.src_disk_mb,
                st.dst_disk_mb,
                st.compression_ratio,
                st.src_ram_mb,
                st.dst_ram_mb
            );
        }
        Err(e) => eprintln!("Erro ao quantizar checkpoint: {}", e),
    }
}
