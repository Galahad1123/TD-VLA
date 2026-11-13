def test_core_import_has_no_gpu_dependencies():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "import td_vla, sys; assert 'torch' not in sys.modules; "
         "assert 'sapien' not in sys.modules"], check=True
    )
