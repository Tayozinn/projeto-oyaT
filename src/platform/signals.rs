//! Tratamento multiplataforma de sinais de interrupção (`Ctrl+C`) sem crates externas.

#[cfg(windows)]
mod ctrlc_imp {
    use std::sync::atomic::{AtomicBool, Ordering};

    type Bool = i32;
    type Dword = u32;
    type HandlerRoutine = unsafe extern "system" fn(Dword) -> Bool;

    const CTRL_C_EVENT: Dword = 0;
    const CTRL_BREAK_EVENT: Dword = 1;
    const TRUE: Bool = 1;
    const FALSE: Bool = 0;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<HandlerRoutine>, add: Bool) -> Bool;
    }

    static GLOBAL_CTRL_C_TRIGGERED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn console_ctrl_handler(ctrl_type: Dword) -> Bool {
        if ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT {
            GLOBAL_CTRL_C_TRIGGERED.store(true, Ordering::SeqCst);
            TRUE
        } else {
            FALSE
        }
    }

    /// Guarda RAII que registra o handler de `Ctrl+C` no console do Windows e o remove no `Drop`.
    pub struct CtrlCGuard {
        registered: bool,
    }

    impl CtrlCGuard {
        pub fn install() -> Self {
            GLOBAL_CTRL_C_TRIGGERED.store(false, Ordering::SeqCst);
            // SAFETY: `console_ctrl_handler` respeita a assinatura `extern "system" fn(u32) -> i32`
            // da API Win32 e realiza apenas escrita atômica em `GLOBAL_CTRL_C_TRIGGERED`.
            let ok = unsafe { SetConsoleCtrlHandler(Some(console_ctrl_handler), TRUE) != 0 };
            Self { registered: ok }
        }

        pub fn is_triggered(&self) -> bool {
            GLOBAL_CTRL_C_TRIGGERED.load(Ordering::SeqCst)
        }
    }

    impl Drop for CtrlCGuard {
        fn drop(&mut self) {
            if self.registered {
                // SAFETY: desregistra o mesmo handler instalado em `install`.
                unsafe {
                    let _ = SetConsoleCtrlHandler(Some(console_ctrl_handler), FALSE);
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod ctrlc_imp {
    /// Guarda RAII de compatibilidade em plataformas não-Windows (sem dependências externas).
    pub struct CtrlCGuard;

    impl CtrlCGuard {
        pub fn install() -> Self {
            Self
        }

        pub fn is_triggered(&self) -> bool {
            false
        }
    }
}

pub use ctrlc_imp::CtrlCGuard;

#[cfg(test)]
mod tests {
    use super::CtrlCGuard;

    #[test]
    fn test_ctrlc_guard_install_and_drop() {
        let guard = CtrlCGuard::install();
        assert!(!guard.is_triggered());
        drop(guard);
    }
}
