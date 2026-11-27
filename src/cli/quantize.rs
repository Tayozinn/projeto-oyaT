use crate::app::quantize::export_quantized_package;

pub(super) fn handle_quantize_command(
    arg: &str,
    base_dir: &std::path::Path,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() < 2 {
        let _ = writeln!(
            out,
            "Uso: :quantize <pasta_ckpt_fp32> <pasta_saida_smq> [fp16|bf16|int8|int4]"
        );
        return;
    }
    let bits_raw = parts.get(2).copied().unwrap_or("int8");
    let prec = match crate::neural::WeightPrecision::parse(bits_raw) {
        Ok(p) => p,
        Err(e) => {
            let _ = writeln!(err, "{}", e);
            return;
        }
    };
    let resolve_rel = |raw: &str| -> std::path::PathBuf {
        let p = std::path::Path::new(raw);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir.join(p)
        }
    };
    let src_dir = resolve_rel(parts[0]);
    let dst_dir = resolve_rel(parts[1]);
    match export_quantized_package(&src_dir, &dst_dir, prec) {
        Ok(st) => {
            let _ = writeln!(
                out,
                "Pacote SMQ1 ({}) gravado em '{}': disco {:.2} MB -> {:.2} MB ({:.2}x menor) | RAM pesos {:.2} MB -> {:.2} MB.",
                st.precision.as_str(),
                dst_dir.display(),
                st.src_disk_mb,
                st.dst_disk_mb,
                st.compression_ratio,
                st.src_ram_mb,
                st.dst_ram_mb
            );
        }
        Err(e) => {
            let _ = writeln!(err, "Erro ao quantizar checkpoint: {}", e);
        }
    }
}
