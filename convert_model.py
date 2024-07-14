import os
import torch
import onnxruntime
import numpy as np
from gnn import create_model
from onnxruntime.datasets import get_example


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def to_onnx(pth_path: str, onnx_path:str):
    if not os.path.exists(pth_path):
        raise FileNotFoundError
    device = 'cpu'
    model = create_model()
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    img = torch.randn(size=(1,3,64,4096))
    pos = torch.randn(size=(1,64,4))
    with torch.no_grad():
        torch_out = model(img, pos)
    # 得到onnx模型的输出
    example_model = get_example(onnx_path) #一定要写绝对路径
    sess = onnxruntime.InferenceSession(example_model)
    onnx_out = sess.run(None,  {"input_img": to_numpy(img),
                                 "input_pos": to_numpy(pos)})

    # 判断输出结果是否一致，小数点后3位一致即可
    np.testing.assert_almost_equal(to_numpy(torch_out), onnx_out[0], decimal=3)
    print("finish")


if __name__ == '__main__':
    weight = ''
    onnx_path = ''
    to_onnx(weight, onnx_path)
