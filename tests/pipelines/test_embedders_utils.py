from pipelines._embedders import _onnx_providers


def test_onnx_providers_for_cpu_and_cuda() -> None:
    assert _onnx_providers("cpu") == ["CPUExecutionProvider"]
    assert _onnx_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert _onnx_providers("cuda:0") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert _onnx_providers("mps") == ["CPUExecutionProvider"]
