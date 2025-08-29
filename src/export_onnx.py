import argparse, torch, os
from .models.fusion import FusionNet

def main(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = FusionNet()
    model.eval()
    face = torch.randn(1,3,112,112)
    audio = torch.randn(1,100,40)
    sensors = torch.randn(1,6,256)
    torch.onnx.export(model, (face,audio,sensors), out_path,
        input_names=['face','audio','sensors'], output_names=['logits'],
        dynamic_axes={'face':{0:'B'},'audio':{0:'B'},'sensors':{0:'B'}}, opset_version=17)
    print('Saved to', out_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='artifacts/fusion_model.onnx')
    args = ap.parse_args()
    main(args.out)