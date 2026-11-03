"""Minimal task used only to test the kernel sandbox."""


def kernel_setup(cfg):
    return {"_wsp_status": lambda: "", "_wsp_finalize": lambda: None}
